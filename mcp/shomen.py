#!/usr/bin/env python
"""Shomen: the deep-thinking context. Study the thing, then answer.

NAMED FROM BONSAI, like `yamadori` and `nebari` and for the same reason. The
shomen is the front of a tree -- the angle it is meant to be seen from. Finding
it is not a step you rush: you turn the tree, study it from every side, and only
then make the first cut. Nobody watching the finished tree sees the turning.

That is exactly this module. It searches, reads and reasons in a context of its
own, and only the distillation is shown. To a user it is never "a second context"
or "a hemisphere" -- it is the system thinking deeply, and every user-facing
string says so.

THE PROBLEM THIS SOLVES, WHICH IS NOT PARALLELISM

Answering "why does this render black in r185 but not r180" takes six searches,
four file reads and a page of reasoning about what came back. Done in the
user's own context, all of it stays there: re-prefilled every turn, occupying
the window that long-context recall degrades over, and ninety percent of it is
scaffolding nobody wanted to read.

Done in a SECOND context, only the distillation crosses back.

WHAT CROSSES BACK (operator decision, 2026-09-23)

"Context saving will never work if the second brain is implementing or
answering a question and never sending back the distillation and facts."
The old rule was "only the conclusion crosses", and the proxy enforced it by
DROPPING: a run that made no search was withheld as general knowledge, and a
run that hit the turn cap without writing a finding handed back nothing
after ten searches. Both threw the work away.

The rule now has two halves:

  - SUMMARIZATION AND LOOKUP work: only the conclusion crosses. The searching
    stays here; that is the saving.
  - WORK THE SECOND BRAIN DID (investigating, answering, implementing): it
    comes back as a DISTILLATION PLUS FACTS, never thrown away, in one fixed
    hand-off (SECTIONS, rendered by `handoff()`):

        FACTS                    each ending path:line if it was read, or
                                 "(reasoning, not checked against source)"
        SEARCHED, FOUND NOTHING  so the main model does not repeat it
        OPEN QUESTIONS
        NEXT STEP

    Every fact crosses. Its citations are checked against what was retrieved
    and the unchecked ones are LABELLED, not dropped. With no search at all
    the whole hand-off crosses under a head that says "reasoning, no sources
    checked". At the turn cap or the helper-budget landing the landing
    prompt (LANDING) requires the hand-off from what was gathered, written by
    the helper itself -- a tool result is never pasted across. If the helper
    writes nothing, `machine_handoff()` builds one from the trace (queries
    run, hit counts, paths retrieved), labelled as machine-built. An empty
    hand-off is impossible. MAX_FINDING_CHARS is the breaker, and a cut says
    so.

THE NUMBERS BELOW WERE MEASURED UNDER THE OLD CONCLUSION-ONLY RULE, with a
free-prose finding "under 200 words". The hand-off is structured, has a
250-word target, and now also crosses on runs that used to cross nothing, so
the main-context figures (8,326 -> 2,410, n=26) and the 531-token crossing
must be RE-MEASURED (bench/context_economy.py) before they are cited for the
current design.

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

So every fact is checked against the paths the investigation actually
retrieved. Cited paths are compared with the tool results, not taken on
trust, and the full trace is kept under a handle the caller can ask for. A
fact that passes stands as read; one that does not crosses LABELLED as
unverified reasoning (until 2026-09-23 the whole finding was refused
instead, and the work was lost). A fact is a claim WITH receipts, or a claim
that says it has none -- never a claim posing as read.

THE ONE RUNNER (operator, 2026-09-24). This module is the second brain:
`run(job, ...)` below runs every job -- investigate, plan, alternative,
tiebreak, fixup -- on the one helper lane, each with a concept seed in its user
message. The investigation's hand-off is folded into main's own turn: the
proxy prefills it as main's reasoning and opens the visible answer with
PHRASES["investigate"].

WHAT IS DELIBERATELY NOT HERE

The decision whether to investigate: since Phase 0.6 (operator, 2026-09-24)
four TRIGGERS decide it (mcp/deep.py) -- the model's own `think_deeply` call,
struggle the proxy detects, a known-hard area, a large task kickoff (the
`plan` job here) -- and mcp/selection.py records the reason. Laya is not
consulted for them. The `delegate_investigation` tool survives only as a
header-forced benchmark arm.
And the client's tools: the second brain never gets client access. What the
harness read is in the conversation already, and a file it would need
becomes the hand-off's NEXT STEP, for main to read with the harness's own
tool.
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
# MAX_HOPS (16) was removed on 2026-09-22 at the operator's instruction: a
# count that ENDED the run hid a lying tool behind a slow failure. Since
# 2026-09-23 a per-run tool-turn limit (tiers.tool_turn_limit: 10, 20 at max)
# LANDS the run instead -- tools withdrawn, finding written -- and the trace
# records it. See _investigate().

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

# NOTHING HERE STATES A BUDGET TO THE MODEL. (Since 2026-09-23 the loop does
# count tool turns -- tiers.tool_turn_limit -- but never tells the model a
# number, and the cap lands the run rather than ending it.)
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
#
# THE HAND-OFF FORMAT (operator decision, 2026-09-23). What crosses back is a
# DISTILLATION PLUS FACTS in four fixed sections, not a free-prose finding.
# The sections are routed by a table rather than described in prose (AGENTS.md
# "Prompting this model": a decision-router table measured 10.7 vs 10.0), and
# the prompt carries no prohibition at all -- the old "Never state anything
# you did not read" became a routing row: a statement that was not read is
# still handed back, labelled as reasoning. The word target is a CHOICE, not
# a measurement; MAX_FINDING_CHARS is the breaker behind it.
HANDOFF_TARGET_WORDS = 250
REASONING_LABEL = "(reasoning, not checked against source)"
WEB_LABEL = "(web)"
SECTIONS =(("facts", "FACTS"),
            ("searched_empty", "SEARCHED, FOUND NOTHING"),
            ("open", "OPEN QUESTIONS"),
            ("next", "NEXT STEP"))

SYSTEM = f"""You are investigating a question about a codebase for another
engineer. You have search tools. The engineer gets your hand-off and nothing
else: not your searches, not your reasoning.

Work in this order:
1. Search for what you need. Read the code before describing it.
2. Stop as soon as you can answer.
3. Write the hand-off.

THE HAND-OFF has exactly these four sections, in this order. Put each item on
its own line, starting with "- ". Aim for under {HANDOFF_TARGET_WORDS} words in total.

FACTS
SEARCHED, FOUND NOTHING
OPEN QUESTIONS
NEXT STEP

Route each thing you know with this table:

| what you have | section | end the line with |
|---|---|---|
| a statement you read in a file | FACTS | the path and line, as src/file.ts:42 |
| a statement you read on a web page | FACTS | the page's URL, then {WEB_LABEL} |
| a statement from a skill | FACTS | its id, as skill:react-19-forms |
| a statement you worked out without reading it | FACTS | {REASONING_LABEL} |
| a search that returned nothing useful | SEARCHED, FOUND NOTHING | the tool and what you searched for |
| something you could not settle | OPEN QUESTIONS | what would settle it |
| what the engineer should do next | NEXT STEP | one or two lines |

Write "- none" under a section that has nothing. Write what the code DOES,
not what you did to find it. If the index does not contain the answer, put
that under OPEN QUESTIONS: an honest gap is useful and a guess is not.

When a mockup, diagram or design sketch would help the answer, draw it with
generate_image. Then look at it with describe_image, passing the url it
returned and a question such as "Does this show <what you asked for>?". If
the picture is off, adjust the prompt and draw it again. Put the markdown
line of the final image under FACTS: the link is the only part of the image
that crosses back. Say in words what the final image shows.

WRITE IN PLAIN ENGINEERING ENGLISH. The hand-off is read by another model,
and may be read by a classifier, so structure it rather than narrate it:

- One idea per line. Split compound sentences.
- Name the thing: write the noun where "it" or "this" would go.
- Active voice with a concrete subject: "the loader caches the plan", not
  "the plan is cached".
- Short sentences. Instead of "X, which suppresses Y", write "X. This
  suppresses Y."
- State a condition as a condition: "If N is zero, the branch returns early."
- Keep identifiers, paths, line numbers and measured values verbatim.

Instead of: "A MouseEvent has none and resolves to a different pointer with no
recorded initial click, which suppresses the synthetic event."
Write: "A MouseEvent has no recorded initial click. It resolves to a different
pointer. This condition suppresses the synthetic event\""""

# What a landing asks for -- the tool-turn cap and the helper-budget stop
# both. It REQUIRES the hand-off from what was gathered: the tool results are
# already in this context, and turning them into FACTS is the helper's job.
# The proxy never pastes raw tool output across.
LANDING = ("Stop searching. Write the hand-off now from what you have "
           "gathered, in the four sections FACTS, SEARCHED, FOUND NOTHING, "
           "OPEN QUESTIONS and NEXT STEP. Turn what the searches returned "
           "into FACTS in your own words, each ending with its path:line. "
           "Put what is still missing under OPEN QUESTIONS.")

# THE PLAN JOB (Phase 0.6, trigger 4: TASK KICKOFF; operator, 2026-09-24).
# A new task whose spec is large sends its PLANNING here: the Octopus pilot's
# V0 steps 1-3 spent 12-14k reasoning tokens each planning on main (#18,
# docs/SELF-IMPROVEMENT-LOG.md), in the context every later step re-reads.
# The second brain writes a compact plan in four fixed sections; the proxy
# prefills it as main's reasoning and main starts acting. Same shape as the
# hand-off: a routing table, no prohibition, a word target that is a CHOICE.
PLAN_TARGET_WORDS = 300
PLAN_SECTIONS = (("files", "FILES"),
                 ("order", "ORDER"),
                 ("decisions", "KEY DECISIONS"),
                 ("risks", "RISKS"))
PLAN_SYSTEM = f"""You are planning a software task for another engineer, who
will carry it out with their own tools. You have search tools for library
source, skills, this project's docs and the web. The engineer gets your plan
and nothing else: not your searches, not your reasoning.

Work in this order:
1. Read the task. Search only for what the plan depends on: an API you are
   unsure of, a version, a library the task names.
2. Write the plan.

THE PLAN has exactly these four sections, in this order. Put each item on its
own line, starting with "- ". Aim for under {PLAN_TARGET_WORDS} words in total.

FILES
ORDER
KEY DECISIONS
RISKS

Route each thing with this table:

| what you have | section | how to write it |
|---|---|---|
| a file to create or change | FILES | the path, then what it holds, in a few words |
| a step | ORDER | one action per line, first step first |
| a choice the task leaves open (library, structure, data format) | KEY DECISIONS | the choice and the reason, with path:line, the URL then {WEB_LABEL}, or {REASONING_LABEL} |
| something likely to go wrong | RISKS | the risk and how to check for it |

Write "- none" under a section that has nothing. Keep names, paths, versions
and commands verbatim. Plain engineering English: one idea per line, the
noun named, active voice."""
PLAN_LANDING = ("Stop searching. Write the plan now from what you have, in "
                "the four sections FILES, ORDER, KEY DECISIONS and RISKS.")

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
    r"|hpp|cpp|rs|py|go|zig|wgsl|glsl|md|h|c)(?![A-Za-z0-9])")
# PHASE 0.6 SOURCES (docs/SELF-IMPROVEMENT-PLAN.md): deep thinking also reads
# the web (mcp/research_tools.py: read_web_page, search_web) and our skill
# store (find_skills). A web fact cites its URL and is labelled "(web)"; a
# skill is cited as skill:<id>. `md` joined _PATH for the knowledge base
# (docs/*.md, find_in_knowledge_base). All three are checked against what
# the run retrieved, exactly as a path is.
_URL = re.compile(r"https?://[^\s<>()\[\]\"'`|]+")
_SKILL_REF = re.compile(r"\bskill:[A-Za-z0-9][\w.-]*")


def _post(path: str, payload: dict, timeout: int = 3600) -> dict:
    """One upstream call to the helper context, through the one door.

    Shaped by mcp/model.py -- the proxy's own effort mapping and budget rule
    at the effort of the REQUEST'S TIER, passed by name in `_effort_tier`
    (default `max`, i.e. xhigh effort) -- so this context can never again
    carry its own budget. By tier NAME, not effort string: since the `xhigh`
    tier (medium effort) was added on 2026-09-23, the string "xhigh" names a
    medium-effort tier, and a hardcoded effort="xhigh" here silently became
    medium. It used to send max_tokens=2500 total, below
    what this model thinks for on an ordinary prompt (docs/CONSTRAINTS.md 8).
    `max_tokens` in the payload is the answer allowance: a tool call, or a
    finding.
    """
    import model
    import tiers
    # role="helper": deep thinking's thinking comes out of ITS 3/8 of the
    # pool (mcp/budget.py), never the conversation's 5/8.
    shaped = model.shape(payload, effort=payload.get("_effort_tier") or "max",
                         role="helper")
    # An EFFORT OVERRIDE on the request (X-Yamadori-Features {"effort": ...},
    # how the domain benchmark holds effort fixed under body tier `max`) wins
    # over the tier name's effort, as it does for the main context. Without
    # this, deep thinking on those arms thought at xhigh while the main
    # context thought at medium (found 2026-09-23 before the domain run).
    eff = payload.get("_effort_override")
    if eff and shaped.get("enable_thinking"):
        shaped["reasoning_effort"] = tiers.safe_effort(eff)
    return model.post(shaped, timeout=timeout)


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
    out |= _urls(text) | {s.lower() for s in _SKILL_REF.findall(text or "")}
    return out


def _urls(text: str) -> set[str]:
    """The URLs a text names, trailing punctuation stripped, lowercased."""
    return {u.rstrip(".,;:!?*_").lower() for u in _URL.findall(text or "")
            if len(u.rstrip(".,;:!?*_")) > len("https://")}


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
    With YAMADORI_E1=1 Laya is off the request path: this raises.
    """
    import e1
    if not e1.laya_allowed():
        raise RuntimeError("Laya is not consulted: YAMADORI_E1=1")
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


# ===================================================== THE ONE RUNNER ======
#
# ONE SECOND-BRAIN RUNNER (operator, 2026-09-24). Three paths used to do
# second-brain work -- this module's investigation, fan-out's B/C
# generation (mcp/fanout.py) and the repair loops (proxy._repair_*, the
# tool-call rounds in mcp/tool_code.py, both of which ran ON MAIN, as extra
# turns in the conversation's own context). They are one runner now, `run`,
# on the one helper lane (admission.helper_lane), with four job types:
#
#   investigate  searches library source with our tools; hands back the four
#                sections (FACTS / SEARCHED, FOUND NOTHING / OPEN / NEXT)
#   alternative  fan-out's candidate B: the same task answered independently
#   tiebreak     fan-out's candidate C: the task plus both candidates and
#                their check results
#   fixup        code that fails the check -- a client write, or a final
#                answer's fenced block: ONLY the code, its errors and the
#                user's request; corrected up to tool_code.REPAIR_ROUNDS
#                rounds at the request's own effort; the last version
#                comes back only if fix_rejection accepts it, else the
#                original. Main never sees a failed attempt.
#
# Every job carries a concept seed in its USER message (operator,
# 2026-09-23 / 2026-09-24). The CALLER supplies it -- proxy.ledger_seed draws
# it once per (request, job) and records it, so a replay of the request uses
# the same word and the job's prompt is byte-identical.
#
# What comes back is folded into main's turn by the proxy in fixed phrases,
# PHRASES below; AGENTS.md carries the same table. A phrase that is
# PREFILLED (the investigate opening) must end on a letter or on an ending
# measured safe, never a space: STEP 0 measured "After thinking deeply," at
# 976 of 995 prompt tokens reused on the next request, twice, and showed that
# a trailing space is its own token (docs/SELF-IMPROVEMENT-PLAN.md).
PHRASES = {
    # prefix of every fold-back that a seeded job produced
    "seed": "Today I was inspired by {words}.",
    # investigate: opens main's visible answer (prefilled); the hand-off is
    # prefilled as main's reasoning
    "investigate": "After thinking deeply,",
    # the check passed: a one-line note, no generation
    "verified": "Verified",
    # fixup changed the code
    "repaired": "Repaired",
    # fan-out: B (and C) were compared with the answer
    "compared": "Compared two approaches",
    # medium: the check found errors and nothing fixes them at this effort
    "checked": "Checked",
}
JOBS = ("investigate", "plan", "alternative", "tiebreak", "fixup")


def seed_line(seeds) -> str:
    """"Today I was inspired by <word>." -- one line per fold-back, naming
    every seed its jobs drew ("X and Y" when B and C both ran); "" when no
    seed was drawn."""
    words = []
    for s in seeds or []:
        w = s.get("word") if isinstance(s, dict) else s
        if w and w not in words:
            words.append(str(w))
    if not words:
        return ""
    joined = (words[0] if len(words) == 1 else
              ", ".join(words[:-1]) + " and " + words[-1])
    return PHRASES["seed"].format(words=joined)


def seed_phrase(word: str, where: str) -> str:
    """The concept seed as every second-brain job's USER message carries it
    (#13, docs/SELF-IMPROVEMENT-LOG.md). The bare "Inspiration word: X"
    (concept_seed.phrase) read like a label to reproduce: live 2026-09-24
    the tie-breaker's seed came back as the first line of the delivered
    code, `# humanidad`. This says what the word is FOR -- the approach --
    and where it does not belong, in one clause naming that failure (the
    prompting rule: at most two prohibitions, each for an observed failure).
    No hedge against its purpose: it is still meant to steer the approach.
    Recorded for the dashboard's last-seed panel as phrase() records it."""
    import concept_seed
    concept_seed.record(word, where)
    return (f"\n\nInspiration word: {word} -- let it shape how you approach "
            f"the task; it is not part of the answer, so it does not appear "
            f"in your code or text.")


def opening(seeds) -> str:
    """The investigate fold-back's visible opening, prefilled as main's
    content. Ends on the measured-safe "After thinking deeply,"."""
    line = seed_line(seeds)
    return (line + " " if line else "") + PHRASES["investigate"]


def run(job: str, *, lane_timeout: float | None = None, held: bool = False,
        **spec) -> dict:
    """Run one second-brain job on the helper lane.

    Returns the job's hand-off: `investigate` -> investigate()'s record
    (`handoff` text and `handoff_stats`); `alternative` / `tiebreak` -> the
    candidate (`content`, `variant`, `seed`); `fixup` -> fixup()'s record
    (per unit the last code and its check). Every result carries `job`, and
    `skipped: "helper busy"` when the lane never came free -- reported, never
    swallowed: work that quietly did not happen looks exactly like work that
    found nothing.

    `held`: the caller already holds the helper lane (fanout.run holds it
    across B and C, so one fan-out's candidates are never split by another
    request's job)."""
    import admission
    import contextlib
    if job not in JOBS:
        raise ValueError(f"unknown second-brain job {job!r}; one of {JOBS}")
    wait = admission.WAIT_SECONDS if lane_timeout is None else lane_timeout
    lane = (contextlib.nullcontext(True) if held else
            admission.helper_lane(timeout=wait, what=job))
    with lane as got:
        if not got:
            return {"job": job, "ok": False, "skipped": "helper busy",
                    "seed": concept_summary(spec.get("seed"))}
        if job in ("investigate", "plan"):
            res = investigate(spec["question"], spec["tools"],
                              spec["run_tool"], spec.get("context", ""),
                              on_think=spec.get("on_think"),
                              tier=spec.get("tier", "max"),
                              effort=spec.get("effort"),
                              seed=spec.get("seed"), mode=job,
                              **({"source_root": spec["source_root"]}
                                 if spec.get("source_root") else {}))
        elif job == "fixup":
            res = fixup(spec["units"], spec.get("request", ""),
                        check=spec["check"], tier=spec.get("tier", "max"),
                        effort=spec.get("effort"), seed=spec.get("seed"))
        else:
            import fanout
            variant = dict(spec["variant"], seed=spec.get("seed"))
            res = fanout._generate(spec["body"], variant,
                                   spec.get("timeout", 3600))
            res["ok"] = bool(res.get("content")) and not res.get("error")
    res["job"] = job
    return res


def concept_summary(seed) -> dict | None:
    import concept_seed
    return concept_seed.summary(seed) if isinstance(seed, dict) else None


FIXUP_SYSTEM = ("You repair code that does not parse. You get the request it "
                "was written for, the code, and the exact errors a parser "
                "reported. Reply with the corrected code in ONE fenced block: "
                "the whole of what you were given, with the smallest change "
                "that makes it parse: change only the lines the errors "
                "require, and keep every other line exactly as given, its "
                "formatting included.")


def _fence(code: str, lang: str) -> str:
    """`code` in a backtick fence one longer than any backtick run at the
    start of one of its lines (at least three), so no line of it can close
    the fence."""
    runs = [len(m.group(1)) for m in _LINE_TICKS.finditer(code or "")]
    fence = "`" * max(3, max(runs, default=0) + 1)
    return f"{fence}{lang}\n{code.rstrip()}\n{fence}"


_LINE_TICKS = re.compile(r"^ {0,3}(`+)", re.M)


def _error_lines(errors: list) -> str:
    out = []
    for e in (errors or [])[:8]:
        if isinstance(e, dict):
            out.append(f"line {e.get('line', 1)}, col {e.get('col', 1)}: "
                       f"{e.get('message', '')}")
        else:
            out.append(str(e))
    return "\n".join(f"- {x}" for x in out) or "- (the parser gave no detail)"


def fixup_prompt(unit: dict, request: str) -> str:
    """The fixup job's user message for one unit: the request, the code and
    its errors. Nothing else from the conversation crosses."""
    lang = unit.get("language") or ""
    where = unit.get("path") or "the answer"
    head = (f"The request:\n{(request or '').strip()[:2000] or '(none)'}\n\n")
    if unit.get("kind") == "edit" and unit.get("old"):
        body = (f"An edit to {where} ({lang}) replaces this text:\n"
                f"{_fence(unit['old'], lang)}\n\nwith this replacement, which "
                f"does not parse:\n{_fence(unit['code'], lang)}\n\n"
                f"Errors:\n{_error_lines(unit.get('errors'))}\n\nWrite the "
                f"corrected replacement.")
    else:
        what = ("file" if unit.get("kind") == "file" else "code block")
        body = (f"This {what} ({where}, {lang}) does not parse:\n"
                f"{_fence(unit['code'], lang)}\n\nErrors:\n"
                f"{_error_lines(unit.get('errors'))}\n\nWrite the corrected "
                f"{what}.")
    return head + body


def _closes_inside(original: str, fence: str) -> bool:
    """True when a line of `original` is a bare run of the fence's character
    at least as long as `fence`: a block opened with that fence was closed
    at that line, so what the reply's block holds is a fragment."""
    ch = (fence or "`")[0]
    for ln in (original or "").split("\n"):
        t = ln.lstrip(" ")
        if (len(ln) - len(t)) <= 3 and t.startswith(fence) \
                and not t.strip().strip(ch):
            return True
    return False


def _largest_block(text: str, original: str = "") -> str | None:
    """The largest CLOSED fenced block in a fix-up reply, or None. An
    unclosed fence is a reply that was cut off, and a block whose fence the
    original code could close (a file containing a bare ``` line, answered
    in a ``` fence) holds a fragment: neither is ever written anywhere
    (pre-deploy review, 2026-09-24)."""
    import code_check
    blocks = [b["code"] for b in code_check.fenced_blocks(text or "")
              if b["code"].strip() and b.get("closed")
              and not _closes_inside(original, b.get("fence") or "```")]
    return max(blocks, key=len) if blocks else None


# The fixed code replaces the model's only when it is plausibly COMPLETE:
# at least this share of the original's non-blank characters. A CHOICE, not
# a measurement (pre-deploy review, 2026-09-24): the job is told to change
# only the lines the errors require, so a minimal fix keeps nearly all of
# the original; a version under half of it is a truncation or a fragment,
# not a fix. No fix-up length distribution exists in this repo yet.
FIXUP_MIN_KEEP = 0.5


def _nonblank(s: str) -> int:
    return len("".join((s or "").split()))


def fix_rejection(original: str, code: str | None, errors: list,
                  why: str | None) -> str | None:
    """Why a fix-up's last version must NOT replace the model's code, or
    None when it may. The rule (pre-deploy review, 2026-09-24): the version
    came from a CLOSED fence (`_largest_block`), the reply was not cut off
    (finish_reason "length"; both surface as `why`), it parses with no
    errors left, and it is plausibly complete (FIXUP_MIN_KEEP)."""
    if why:
        return why
    if code is None:
        return "no version came back"
    if errors:
        return (f"{len(errors)} error{'s' if len(errors) != 1 else ''} "
                f"still in the repaired version")
    if _nonblank(code) < FIXUP_MIN_KEEP * _nonblank(original):
        return (f"the repaired version is under {int(FIXUP_MIN_KEEP * 100)}% "
                f"of the original's length (a fragment or a truncation)")
    return None


def fixup(units: list[dict], request: str = "", *, check, tier: str = "max",
          effort: str | None = None, seed: dict | None = None) -> dict:
    """Correct each unit's code, up to tool_code.REPAIR_ROUNDS rounds.

    `units`: [{path, language, kind (file|edit|block), code, old?, errors}].
    `check(unit, code) -> errors` re-checks a version (the caller's checker:
    tool_code for a client write, code_check for an answer's block).
    Returns {ok, units: [{..., code, errors_before, errors_after, rounds,
    changed, accepted, rejected}], rounds, seed}. `code` is the repaired
    version ONLY when `fix_rejection` finds nothing against it; otherwise it
    is the unit's original code, untouched, `errors_after` counts the
    original's errors (what would be sent), and `rejected` says why."""
    import concept_seed
    import tiers
    import tool_code
    out, most = [], 0
    for u in units:
        n = 0
        cand, errs = None, list(u.get("errors") or [])
        why = None
        convo = [{"role": "system", "content": FIXUP_SYSTEM},
                 {"role": "user", "content": fixup_prompt(u, request) + (
                     seed_phrase(seed["word"], where="fixup")
                     if seed else "")}]
        echoed = 0
        while errs and n < tool_code.REPAIR_ROUNDS:
            n += 1
            payload = {"model": MODEL, "messages": convo, "tools": [],
                       "max_tokens": max(tiers.A_MIN,
                                         len(cand or u["code"]) // 2 + 512),
                       "_effort_tier": tier, "_effort_override": effort}
            try:
                d = _post("/v1/chat/completions", payload)
                ch = d["choices"][0]
                reply = (ch["message"].get("content") or "")
                fin = ch.get("finish_reason")
            except Exception as e:                               # noqa: BLE001
                why = f"{type(e).__name__}: {e}"[:200]
                break
            if fin == "length":
                # A budget event, never an answer (model.BudgetEvent).
                why = "the reply was cut off (finish_reason length)"
                break
            code = _largest_block(reply, u["code"])
            if code is None:
                why = "the reply carried no closed fenced block"
                break
            # The seed copied into the code as its first line (#13): this
            # code is written into the client's file.
            import fanout
            code, k = fanout.strip_seed_echo(code, (seed or {}).get("word"))
            echoed += k
            # A fence drops the file's last newline; keep the original's.
            if u["code"].endswith("\n") and not code.endswith("\n"):
                code += "\n"
            cand, errs = code, list(check(u, code) or [])
            if errs:
                convo += [{"role": "assistant", "content": reply},
                          {"role": "user", "content":
                           "It still does not parse:\n" + _error_lines(errs)
                           + "\n\nWrite the corrected code again, whole, in "
                             "one fenced block."}]
        most = max(most, n)
        rejected = fix_rejection(u["code"], cand, errs, why)
        before = list(u.get("errors") or [])
        out.append(dict(u, code=u["code"] if rejected else cand,
                        errors_after=len(before) if rejected else 0,
                        errors_left=before if rejected else [],
                        candidate_errors=len(errs) if cand is not None
                        else None,
                        rounds=n, errors_before=len(before),
                        accepted=not rejected, rejected=rejected,
                        changed=not rejected and cand != u["code"], why=why,
                        seed_echo_stripped=echoed))
    return {"ok": all(x["accepted"] for x in out), "units": out,
            "rounds": most, "seed": concept_seed.summary(seed)}


def investigate(question: str, tools: list[dict], run_tool,
                context: str = "", hops: int | None = None,
                on_think=None, tier: str = "max",
                effort: str | None = None,
                seed: dict | None | bool = True,
                mode: str = "investigate",
                source_root: str | None = None) -> dict:
    """Run a tool loop in a private context and return its hand-off.

    `mode` "plan" (Phase 0.6 task kickoff) runs the same loop with
    PLAN_SYSTEM and hands back the four PLAN_SECTIONS (plan_handoff).

    The result's `handoff` (also `finding`) is the fixed four-section text
    and `handoff_stats` its counts for x_yamadori; see the module docstring,
    WHAT CROSSES BACK.

    `run_tool(name, args) -> str` is injected rather than imported so this can
    be exercised with a fake in tests, and so the caller decides which index
    and which repository the investigation is scoped to.

    EVERY second-brain run carries a concept seed (operator, 2026-09-23),
    put in the USER turn with the question (concept_seed.phrase, which also
    records it for the dashboard's last-seed panel). `seed` is the one the
    caller recorded (proxy.ledger_seed, via run()); True draws a fresh one,
    away from the question. The result's `seed` is {word, token_id, u32}, or
    None when the embedding matrix is not extracted.
    """
    import concept_seed
    if seed is True:
        # A direct caller (a script, the delegate arm's old path) draws its
        # own; the proxy passes the one its ledger recorded (run()).
        seed = (concept_seed.seed_for(question, 1) or [None])[0]
    res = _investigate(question, tools, run_tool, context, hops, on_think,
                       seed, tier=tier, effort=effort, mode=mode,
                       source_root=source_root)
    res["mode"] = mode
    res["seed"] = concept_seed.summary(seed)
    res["effort_tier"] = tier
    # The effort string actually sent upstream: the override if there was
    # one, else the tier's own (mirrors _post). Recorded so a row can show
    # what deep thinking thought at.
    import tiers
    res["effort"] = tiers.safe_effort(
        effort or tiers.resolve({"reasoning_effort": tier})["effort"])
    return res


def _investigate(question: str, tools: list[dict], run_tool, context: str,
                 hops: int | None, on_think, seed: dict | None,
                 tier: str = "max", effort: str | None = None,
                 mode: str = "investigate",
                 source_root: str | None = None) -> dict:
    started = time.time()
    plan = mode == "plan"
    landing = PLAN_LANDING if plan else LANDING
    convo = [{"role": "system", "content": PLAN_SYSTEM if plan else SYSTEM}]
    if context:
        convo.append({"role": "user",
                      "content": f"Context from the conversation:\n{context[:1500]}"})
    convo.append({"role": "user", "content": question + (
        seed_phrase(seed["word"], where="shomen") if seed else "")})

    trace: list[dict] = []
    seen_paths: set[str] = set()
    spent = 0

    # The old hop count (MAX_HOPS 16) was removed 2026-09-22. This context
    # ends when it stops asking for tools, when its window reaches its share
    # of the KV pool (the helper-budget check below), or at the vendor's
    # tool-turn cap, tiers.tool_turn_limit(tier) -- 10, 20 at max -- counted
    # per deep-thinking run (operator, 2026-09-23; PrismML's
    # agenticMaxTurns = 10): after that many tool turns the tools are
    # withdrawn and it writes up what it has. The trace records the cap as
    # "(turn cap)". `hops`, when a test passes it, bounds generations instead.
    import tiers
    limit = tiers.tool_turn_limit(tier)
    hop = -1
    while True:
        hop += 1
        if hops is not None:
            last = hop >= hops - 1
        else:
            last = hop >= limit
            if last:
                trace.append({"tool": "(turn cap)", "args": {},
                              "result": f"{limit} tool turns reached"})
        if last:
            # GRACEFUL DEGRADATION OF A TRIPPED BREAKER, and nothing else.
            # This is no longer part of a normal run -- a healthy
            # investigation stops asking for tools and never gets here.
            #
            # It is kept because a breaker that simply expires returns
            # nothing for everything it spent, while one that asks for the
            # finding returns whatever was actually found. Withdrawing the
            # tools makes the request unambiguous.
            convo.append({"role": "user", "content": landing})
        payload = {"model": MODEL, "messages": convo,
                   "tools": [] if last else tools,
                   "max_tokens": 1500, "temperature": 0.2,
                   "_effort_tier": tier, "_effort_override": effort}
        try:
            d = _post("/v1/chat/completions", payload)
        except Exception as e:                                   # noqa: BLE001
            return _fail(f"{type(e).__name__}: {e}", trace, started, spent,
                         question, seen_paths)
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
                          "investigation. " + landing})
            try:
                d = _post("/v1/chat/completions",
                          dict(payload, tools=[], messages=convo))
                spent += int((d.get("usage") or {}).get("total_tokens") or 0)
                msg = d["choices"][0]["message"]
            except Exception as e:                               # noqa: BLE001
                return _fail(f"{type(e).__name__}: {e}", trace, started, spent,
                             question, seen_paths)
            return _finish((msg.get("content") or "").strip(), trace,
                           seen_paths, started, spent, question, mode,
                           source_root)

        calls = msg.get("tool_calls") or []

        if not calls:
            finding = (msg.get("content") or "").strip()
            if not finding and msg.get("reasoning_content"):
                # Budget exhausted before writing. Reported as such rather
                # than returned as an empty conclusion, which would read to
                # the caller as "investigated, found nothing".
                return _fail("ran out of budget while reasoning, no conclusion",
                             trace, started, spent, question, seen_paths)
            return _finish(finding, trace, seen_paths, started, spent,
                           question, mode, source_root)
        if last:
            # The landing went out with no tools and still came back asking
            # for one. Nothing would read its result, and looping again would
            # never end (the landing condition stays true), so this stops.
            return _fail(
                f"tool-turn cap reached ({limit if hops is None else hops}) "
                f"and the landing still asked for a tool instead of writing "
                f"the finding -- this indicates a tool returning results the "
                f"model cannot act on, not a budget being exhausted; check "
                f"the trace", trace, started, spent, question, seen_paths)

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
            import repeats
            found = _cited(out)
            seen_paths |= found
            # `empty` and `paths` feed the hand-off: an empty search is listed
            # under SEARCHED, FOUND NOTHING so the main model does not repeat
            # it, and a machine-built hand-off reports hit counts from them.
            trace.append({"hop": hop, "tool": fn, "args": args,
                          "chars": len(out), "empty": repeats._empty(out),
                          "paths": len(found),
                          # The package versions it read: the hand-off's
                          # evidence is read from these (resolve_source).
                          "versions": versions_in(out),
                          "ms": round((time.time() - t0) * 1000)})
            # Same breaker as the proxy's loops, with a marker and the next
            # range to request -- a silent cut handed this context half a file
            # under a header promising all of it (docs/CONSTRAINTS.md 12).
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
        trace, started, spent, question, seen_paths)


def _finish(finding: str, trace: list, seen: set[str], started: float,
            spent: int, question: str = "", mode: str = "investigate",
            source_root: str | None = None) -> dict:
    """Attach the receipts, and hand back what was written -- always.

    A conclusion citing nothing is indistinguishable from a conclusion the
    model invented, and the caller cannot tell the difference because it never
    saw the searches. Checking the citations against what was actually
    retrieved is the only thing standing between context economy and confident
    fabrication.

    Until 2026-09-23 that check REFUSED: an uncited finding crossed with a
    warning, or (in the proxy) not at all. Now it LABELS, per fact
    (`handoff()`): a fact whose cited paths were retrieved stands as read; any
    other fact crosses marked as reasoning or as citing a path this
    investigation never retrieved. Nothing the helper wrote is thrown away,
    and nothing unchecked passes as checked. An empty write-up becomes a
    machine-built hand-off from the trace.
    """
    claimed = _cited(finding)
    supported = {p for p in claimed if _supported(p, seen)}
    unsupported = sorted(claimed - supported)

    handle = uuid.uuid4().hex[:8]
    if len(_TRACES) >= MAX_TRACES:
        _TRACES.pop(next(iter(_TRACES)))
    _TRACES[handle] = {"trace": trace, "finding": finding,
                       "retrieved": sorted(seen)}

    if finding.strip() and mode == "plan":
        text, stats = plan_handoff(finding, trace, seen, handle,
                                   root=source_root)
    elif finding.strip():
        text, stats = handoff(finding, trace, seen, handle, root=source_root)
    else:
        text, stats = machine_handoff(question, trace, seen,
                                      "it returned no text", handle)
    out = {"ok": not stats["machine_built"], "finding": text,
           "handoff": text, "handoff_stats": stats,
           "handle": handle, "hops": len(trace),
           "searches": _searches(trace),
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


def _fail(why: str, trace: list, started: float, spent: int,
          question: str = "", seen: set[str] | None = None) -> dict:
    # The trace is kept on failure TOO. A failed investigation is exactly the
    # one worth inspecting, and the first live run discarded its own evidence
    # because only the success path wrote to the store.
    #
    # And since 2026-09-23 a failure still HANDS OFF: a turn cap whose landing
    # asked for a tool, or a budget spent reasoning, used to return
    # "investigation failed" and the proxy dropped it -- ten searches' results
    # discarded. The machine-built hand-off lists what was run and retrieved.
    seen = set(seen or ())
    handle = uuid.uuid4().hex[:8]
    if len(_TRACES) >= MAX_TRACES:
        _TRACES.pop(next(iter(_TRACES)))
    _TRACES[handle] = {"trace": trace, "finding": why,
                       "retrieved": sorted(seen), "failed": True}
    text, stats = machine_handoff(question, trace, seen,
                                  f"investigation failed: {why}", handle)
    return {"ok": False, "finding": text, "handoff": text,
            "handoff_stats": stats, "error": why,
            "handle": handle, "hops": len(trace),
            "searches": _searches(trace),
            "tools_used": sorted({t["tool"] for t in trace}),
            "cited": [], "unsupported": [], "helper_tokens": spent,
            "seconds": round(time.time() - started, 1)}


# ---------------------------------------------------------------------------
# THE HAND-OFF: what crosses back, as sections rather than prose.
#
# The helper writes the four SECTIONS; this reads them back, checks every
# fact's citations against what was retrieved, labels what is not checked,
# adds the empty searches the trace recorded, and renders the one fixed
# format. It never pastes a tool result: FACTS are the helper's own words, or
# (machine-built) one line of log per search.
#
# Per-section item caps keep it distilled. They are CHOICES, not
# measurements; MAX_FINDING_CHARS is the breaker behind them and says so
# when it cuts.
# ---------------------------------------------------------------------------
MAX_ITEMS = {"facts": 12, "searched_empty": 8, "open": 5, "next": 3}

_SECTION_PATTERNS = {
    "facts": r"facts?|findings?",
    "searched_empty": (r"(?:what was )?searched(?:,| and)?\s+(?:but\s+)?"
                       r"found nothing|searched[- ]empty|nothing found"),
    "open": r"open questions?|unknowns?",
    "next": r"next steps?|recommendations?(?:\s*/\s*next steps?)?",
}
# A heading is a line holding ONLY the section name (markdown hashes, bold
# and a trailing colon allowed), or the name, a colon and an inline item.
# "Facts about the loader" is prose, not a heading.
_HEADING = re.compile(
    r"^\s*(?:#+\s*)?\**\s*(" + "|".join(_SECTION_PATTERNS.values())
    + r")\s*\**\s*(?::\s*\**\s*(.*))?$", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_NONE = re.compile(r"^\(?\s*none\.?\s*\)?$", re.IGNORECASE)


def _supported(path: str, seen: set[str]) -> bool:
    return any(path == s or s.endswith("/" + path) or path.endswith("/" + s)
               for s in seen)


def _searches(trace: list) -> int:
    """Real tool calls in a trace -- not the "(turn cap)" or "(budget)"
    markers, which are trace entries but searched nothing."""
    return sum(1 for t in trace if "hop" in t)


def _section_of(name: str) -> str:
    for key, pat in _SECTION_PATTERNS.items():
        if re.fullmatch(pat, name.strip(), re.IGNORECASE):
            return key
    return "facts"


def parse_sections(text: str) -> tuple[dict, bool]:
    """{section: [item, ...]} from a written hand-off, and whether any
    heading was found. Text before the first heading, or a whole write-up
    with none, is read as FACTS: an unstructured finding is still handed
    back, one fact per bullet or paragraph, and each one is checked."""
    out: dict = {k: [] for k, _ in SECTIONS}
    cur, open_item, structured = "facts", False, False
    in_fence = False
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if not in_fence:
            m = _HEADING.match(line)
            if m:
                cur, structured = _section_of(m.group(1)), True
                open_item = False
                rest = (m.group(2) or "").strip().strip("*").strip()
                if rest and not _NONE.match(rest):
                    out[cur].append(rest)
                    open_item = True
                continue
        if not line.strip():
            open_item = in_fence and open_item
            continue
        if not in_fence and _BULLET.match(line):
            item = _BULLET.sub("", line, count=1).strip()
            if not _NONE.match(item):
                out[cur].append(item)
                open_item = True
            else:
                open_item = False
            continue
        if open_item and out[cur]:
            # A continuation keeps its line break, so code in a fact survives.
            out[cur][-1] += "\n  " + line.strip()
        else:
            out[cur].append(line.strip())
            open_item = True
    return out, structured


# ---------------------------------------------------------------------------
# THE EVIDENCE (operator, 2026-09-24). Main and the user cannot open a path
# on this server: `three/src/core/Object3D.js:714` names a file in OUR package
# index, so a bare citation is useless to them. Every VERIFIED fact carries
# its evidence inline -- the lines it cites, read from the held source by the
# verifier itself (never the helper's memory), labelled with the package and
# version they came from:
#
#     - lookAt flips the target for cameras. three/src/core/Object3D.js:714
#       source: three@0.185.1 src/core/Object3D.js:711-717
#     ```js
#     ...the lines, exactly as the file holds them...
#     ```
#
# And a citation the verifier cannot read -- no such held file, or a line the
# file does not have -- is REMOVED from the fact, which crosses relabelled
# REASONING_LABEL (live gate 2026-09-24: "three/src/core/Open/Three.js:714"
# crossed although no such file is held). The caps are CHOICES, unmeasured.
EXCERPT_MAX_LINES = 12          # one fact's excerpt
EXCERPT_CONTEXT = 3             # lines either side of a single cited line
EXCERPT_TOTAL_CHARS = int(os.environ.get("YAMADORI_EXCERPT_CHARS", "3600"))
_EXTS = ("tsx|ts|jsx|js|mjs|cjs|json|jsonc|toml|yaml|yml|hpp|cpp|rs|py|go"
         "|zig|wgsl|glsl|md|h|c")
# A file citation with its optional line or range, as the helper writes it.
_FILE_CITE = re.compile(
    r"(?<![\w@./\\-])((?:\.{1,2}/)?[\w@.\\/-]*?[\w-]+\.(?:" + _EXTS + r"))"
    r"(?![A-Za-z0-9])(?::(\d+)(?:\s*[-–]\s*(\d+))?)?")
_LANG = {"ts": "ts", "tsx": "tsx", "js": "js", "jsx": "jsx", "mjs": "js",
         "cjs": "js", "py": "python", "rs": "rust", "go": "go",
         "wgsl": "wgsl", "glsl": "glsl", "json": "json", "md": "markdown",
         "yaml": "yaml", "yml": "yaml", "toml": "toml", "c": "c", "h": "c",
         "cpp": "cpp", "hpp": "cpp", "zig": "zig", "jsonc": "json"}
# "== three@0.185.1 ==" / "== searched three@0.185.1, ..." in a package
# tool's result: the version the investigation actually read.
_VERSION_HEAD = re.compile(r"==\s*(?:searched\s+)?(@?[\w./-]+?)@(\d[\w.+-]*)")


def versions_in(text: str) -> dict[str, str]:
    """{package: version} a tool result says it searched or read."""
    out: dict[str, str] = {}
    for pkg, ver in _VERSION_HEAD.findall(text or ""):
        out.setdefault(pkg, ver)
    return out


_FILES_CACHE: dict[str, dict[str, str]] = {}


def _package_files(db: str) -> tuple[str, dict[str, str]]:
    """(source root, {lowercased relative path: relative path}) of one held
    package index. Cached per index file."""
    import sqlite3
    if db not in _FILES_CACHE:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            roots = [r[0] for r in con.execute("SELECT path FROM roots")]
            files = {r[0].lower(): r[0] for r in con.execute(
                "SELECT path FROM package_files")}
        finally:
            con.close()
        _FILES_CACHE[db] = {"\x00root": roots[0] if roots else "", **files}
    got = _FILES_CACHE[db]
    return got["\x00root"], got


def _match_file(files: dict[str, str], rel: str) -> str | None:
    """The indexed path `rel` names: exact, else a suffix either way at a
    path boundary -- never a bare file name matched against a deeper
    path (index.js is in every package)."""
    r = rel.lower()
    if r in files:
        return files[r]
    if "/" not in r:
        return None
    for low, real in files.items():
        if low.startswith("\x00"):
            continue
        if r.endswith("/" + low) or low.endswith("/" + r):
            return real
    return None


def resolve_source(path: str, prefer: dict | None = None,
                   root: str | None = None) -> dict | None:
    """Where a cited path lives in the source this server holds:
    {label, rel, file} -- `label` "three@0.185.1" (or "repository") -- or
    None when no held file matches. A path that starts with a held package
    ("three/src/...", "three@0.185.1/src/...", "node_modules/three/...")
    is looked up in that package only; any other path in the packages the
    investigation read (`prefer`, its {package: version}) first, then every
    held package, then the bound repository `root`."""
    import domains
    p = path.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    if p.startswith("node_modules/"):
        p = p[len("node_modules/"):]
    prefer = prefer or {}
    try:
        held = domains.held_sources() or {}
    except Exception:                                            # noqa: BLE001
        held = {}
    names = sorted(held, key=len, reverse=True)
    routed = None
    for pkg in names:
        pl = pkg.lower()
        lp = p.lower()
        if lp.startswith(pl + "/") or lp.startswith(pl + "@"):
            rest = p[len(pkg):]
            ver = None
            if rest.startswith("@"):
                ver, _, rest = rest[1:].partition("/")
            routed = (pkg, ver, rest.lstrip("/"))
            break

    def versions(pkg: str, ver: str | None) -> list:
        rows = list(held.get(pkg) or [])
        want = ver or prefer.get(pkg)
        return sorted(rows, key=lambda r: r[0] != want)   # stable: newest next

    if routed:
        order = [(routed[0], routed[1], routed[2])]
    else:
        pkgs = [k for k in prefer if k in held] + [
            k for k in sorted(held) if k not in prefer]
        order = [(k, None, p) for k in pkgs]
    for pkg, ver, rel in order:
        if not rel:
            continue
        for v, db in versions(pkg, ver):
            try:
                src, files = _package_files(db)
            except Exception:                                    # noqa: BLE001
                continue
            real = _match_file(files, rel)
            if real and src:
                return {"label": f"{pkg}@{v}", "rel": real,
                        "file": os.path.join(src, real)}
    if root and not routed:
        full = os.path.realpath(os.path.join(root, p))
        base = os.path.realpath(root)
        if (os.path.commonpath([full, base]) == base
                and os.path.isfile(full)):
            return {"label": "repository", "rel": p, "file": full}
    return None


def directory_resolver(base: str, label: str = "fixture@0.0.0"):
    """A SOURCE_RESOLVER over one directory, for tests: a cited path is the
    file of that relative path under `base`, or unreadable."""
    def resolve(path: str, prefer: dict | None = None,
                root: str | None = None) -> dict | None:
        p = path.replace("\\", "/").strip()
        while p.startswith("./"):
            p = p[2:]
        p = p.lstrip("/")
        full = os.path.join(base, p)
        return ({"label": label, "rel": p, "file": full}
                if p and os.path.isfile(full) else None)
    return resolve


# Tests replace this; the default reads the held package indexes.
SOURCE_RESOLVER = resolve_source
_LINES_CACHE: dict[str, list[str]] = {}


def _file_lines(file: str) -> list[str] | None:
    if file not in _LINES_CACHE:
        try:
            with open(file, encoding="utf-8", errors="replace",
                      newline="") as f:
                text = f.read()
        except OSError:
            return None
        if len(_LINES_CACHE) > 64:
            _LINES_CACHE.pop(next(iter(_LINES_CACHE)))
        _LINES_CACHE[file] = [ln.rstrip("\r") for ln in text.split("\n")]
    return _LINES_CACHE[file]


def _evidence_of(fact: str, ev: dict) -> tuple[str, list[dict], list[str]]:
    """(the fact with every citation the verifier cannot read removed, the
    readable line citations [{src, a, b}], the removed citations)."""
    good: list[dict] = []
    bad: list[str] = []
    urls = [(u.start(), u.end()) for u in _URL.finditer(fact)]
    for m in list(_FILE_CITE.finditer(fact)):
        path, a, b = m.group(1), m.group(2), m.group(3)
        if any(s <= m.start() and m.end() <= e for s, e in urls):
            continue                     # part of a URL, not a file
        if a is None and "/" not in path.replace("\\", "/"):
            continue     # a bare file name in prose, not a citation
        src = ev["resolve"](path, ev.get("prefer"), ev.get("root"))
        lines = _file_lines(src["file"]) if src else None
        ok = src is not None and lines is not None
        if ok and a is not None:
            ai, bi = int(a), int(b) if b else None
            n = len(lines)
            ok = 1 <= ai <= n and (bi is None or ai <= bi <= n)
            if ok:
                good.append({"src": src, "a": ai, "b": bi, "text": m.group(0)})
        if not ok:
            bad.append(m.group(0))
    if not bad:
        return fact, good, []
    out = fact
    for cite in bad:
        out = out.replace(cite, "")
    # What removing a citation leaves behind: "()", "``", " ,", doubled
    # spaces, a dangling "at" or "in".
    out = re.sub(r"\(\s*\)|`\s*`|\[\s*\]", "", out)
    out = re.sub(r"\s+(?:at|in|see|from)\s*([.,;:]?)\s*$", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    out = re.sub(r"[\s,;:]+$", "", out)
    if not out:
        out = ("a fact whose only content was a citation the verifier could "
               "not read (removed)")
    return out, good, bad


def _excerpt(cite: dict) -> tuple[str, str]:
    """(label, code) of one readable citation: the cited range (at most
    EXCERPT_MAX_LINES), or EXCERPT_CONTEXT lines either side of one line."""
    lines = _file_lines(cite["src"]["file"]) or []
    a, b = cite["a"], cite["b"]
    if b is None:
        start = max(1, a - EXCERPT_CONTEXT)
        end = min(len(lines), a + EXCERPT_CONTEXT)
    else:
        start, end = a, min(b, a + EXCERPT_MAX_LINES - 1)
    # The label and the body must name the SAME lines (live gate
    # 2026-09-24, second run: 3 of 5 excerpts ended on a blank line, which
    # the fence dropped, so the body was one line short of its label). Blank
    # lines at either edge are left out of the range, and the label says so.
    while end > start and not lines[end - 1].strip():
        end -= 1
    while start < end and not lines[start - 1].strip():
        start += 1
    code = "\n".join(lines[start - 1:end])
    return (f"{cite['src']['label']} {cite['src']['rel']}:{start}-{end}",
            code)


def _with_excerpt(fact: str, cite: dict, ev: dict) -> str:
    """The fact with its evidence inline, within the hand-off's excerpt
    budget (ev["left"]); past it, the fact says the excerpt was not
    inlined, rather than crossing as if it had none to show."""
    label, code = _excerpt(cite)
    lang = _LANG.get(cite["src"]["rel"].rsplit(".", 1)[-1].lower(), "")
    # Not _fence(): it strips the code's trailing whitespace, and an excerpt
    # is the file's lines EXACTLY.
    runs = [len(m.group(1)) for m in _LINE_TICKS.finditer(code)]
    ticks = "`" * max(3, max(runs, default=0) + 1)
    block = f"\n  source: {label}\n{ticks}{lang}\n{code}\n{ticks}"
    if len(block) > ev["left"]:
        ev["skipped"] += 1
        return (f"{fact} [excerpt not inlined: the hand-off's excerpt "
                f"budget, {EXCERPT_TOTAL_CHARS} characters (a choice), is "
                f"spent]")
    ev["left"] -= len(block)
    ev["excerpts"] += 1
    ev["chars"] += len(block)
    return fact + block


def _evidence_state(trace: list | None = None, root: str | None = None
                    ) -> dict:
    prefer: dict[str, str] = {}
    for t in trace or []:
        for k, v in (t.get("versions") or {}).items():
            prefer.setdefault(k, v)
    return {"resolve": SOURCE_RESOLVER, "prefer": prefer, "root": root,
            "left": EXCERPT_TOTAL_CHARS, "excerpts": 0, "chars": 0,
            "skipped": 0, "removed": 0}


def _label_fact(fact: str, seen: set[str],
                ev: dict | None = None) -> tuple[str, bool]:
    """(the fact as it crosses, verified?). Verified means it cites at least
    one path, every path it cites was retrieved by this investigation, and
    every file citation is one the verifier could read in the held source
    -- whose lines then cross with it (THE EVIDENCE, above). A citation the
    verifier cannot read is removed, and the fact crosses as reasoning."""
    ev = ev if ev is not None else _evidence_state()
    fact, good, bad = _evidence_of(fact, ev)
    ev["removed"] += len(bad)
    low = fact.lower()
    if "reasoning" in low and "not checked" in low:
        return fact, False
    if bad:
        return f"{fact} {REASONING_LABEL}", False
    cited = _cited(fact)
    if not cited:
        return f"{fact} {REASONING_LABEL}", False
    # A fact read on the web says so (operator, 2026-09-24): its URL is the
    # citation and "(web)" the label, whether or not the helper wrote it.
    if _urls(fact) and WEB_LABEL not in fact:
        fact = f"{fact} {WEB_LABEL}"
    missing = sorted(p for p in cited if not _supported(p, seen))
    if missing:
        return (f"{fact} [cites {', '.join(missing[:3])}, which this "
                f"investigation did not retrieve: unverified]"), False
    line_cites = [c for c in good if c["a"] is not None]
    if line_cites:
        fact = _with_excerpt(fact, line_cites[0], ev)
    return fact, True


def _describe(step: dict) -> str:
    try:
        import streaming
        return streaming.describe_call(step.get("tool") or "",
                                       step.get("args") or {})
    except Exception:                                            # noqa: BLE001
        return str(step.get("tool") or "")


def _render(sections: dict, lead: str = "", handle: str = "") -> str:
    parts = [lead] if lead else []
    for key, title in SECTIONS:
        items = sections.get(key) or []
        cap = MAX_ITEMS[key]
        lines = [f"- {i}" for i in items[:cap]] or ["- none"]
        if len(items) > cap:
            lines.append(f"- (+{len(items) - cap} more"
                         + (f"; trace handle {handle}" if handle else "")
                         + ")")
        parts.append(title + "\n" + "\n".join(lines))
    return "\n".join(parts)


def _cut(text: str, extra: int = 0) -> tuple[str, bool]:
    """MAX_FINDING_CHARS is the breaker on the helper's own words; the
    inlined evidence (`extra`, itself capped at EXCERPT_TOTAL_CHARS) rides
    on top of it."""
    limit = MAX_FINDING_CHARS + max(0, extra)
    if len(text) <= limit:
        return text, False
    return (text[:limit]
            + f"\n\n[hand-off cut at {limit} characters of "
              f"{len(text)}]"), True


def _stats(sections: dict, verified: int, text: str, machine: bool,
           structured: bool, cut: bool, ev: dict | None = None) -> dict:
    facts = len(sections.get("facts") or [])
    out = {"facts": facts, "verified": verified,
           "unverified": facts - verified if not machine else 0,
           "searched_empty": len(sections.get("searched_empty") or []),
           "open": len(sections.get("open") or []),
           "chars": len(text), "machine_built": machine,
           "structured": structured, "cut": cut}
    if ev is not None:
        # THE EVIDENCE: excerpts inlined, their characters, verified facts
        # whose excerpt the budget left out, and citations removed because
        # the verifier could not read them.
        out.update(excerpts=ev["excerpts"], excerpt_chars=ev["chars"],
                   excerpts_skipped=ev["skipped"],
                   citations_removed=ev["removed"])
    return out


def handoff(text: str, trace: list, seen: set[str],
            handle: str = "", root: str | None = None) -> tuple[str, dict]:
    """The helper's write-up as the fixed hand-off, and its x_yamadori stats.

    Every fact is kept. Its citations are checked against `seen` (the paths
    the investigation's tool results named) and read from the held source
    (THE EVIDENCE): verified facts stand as read, with the lines they cite
    inlined; the rest cross labelled, and a citation that cannot be read is
    removed. With no search at all every fact is reasoning. The trace's
    empty searches are added under SEARCHED, FOUND NOTHING, so the main
    model is told what not to repeat even when the helper forgot.
    `root`: the bound repository, for a citation of one of its files.
    """
    sections, structured = parse_sections(text)
    ev = _evidence_state(trace, root)
    facts, verified = [], 0
    for f in sections["facts"][:MAX_ITEMS["facts"]]:
        labelled, ok = _label_fact(f, seen, ev)
        facts.append(labelled)
        verified += ok
    # Past the cap a fact is not shown (_render), so it is labelled without
    # spending the excerpt budget on it.
    for f in sections["facts"][MAX_ITEMS["facts"]:]:
        labelled, ok = _label_fact(f, seen, dict(ev, left=0))
        facts.append(labelled)
        verified += ok
    sections["facts"] = facts
    # OPEN QUESTIONS and NEXT STEP are not checked: they name what is NOT
    # known, and a NEXT STEP may name a file of the user's own project for
    # main to read with the harness's tools (Phase 0.5).
    have = {s.lower() for s in sections["searched_empty"]}
    for t in trace:
        if t.get("empty") and "hop" in t:
            d = _describe(t)
            if not any(d.lower() in h or h in d.lower() for h in have):
                sections["searched_empty"].append(d + " (search log)")
                have.add(d.lower())
    rendered, cut = _cut(_render(sections, handle=handle), extra=ev["chars"])
    return rendered, _stats(sections, verified, rendered, False, structured,
                            cut, ev)


def machine_handoff(question: str, trace: list, seen: set[str], why: str,
                    handle: str = "") -> tuple[str, dict]:
    """The hand-off the PROXY builds when the helper wrote none.

    An empty hand-off is impossible (operator, 2026-09-23): whatever the
    investigation ran and retrieved crosses back, labelled as machine-built,
    as log lines -- the queries run, their hit counts, the paths retrieved.
    It states no conclusion, because none was written.
    """
    steps = [t for t in trace if "hop" in t]
    facts = []
    for t in steps:
        if t.get("empty"):
            continue
        n = t.get("paths") or 0
        facts.append(f"{_describe(t)} returned {t.get('chars', 0)} "
                     f"characters" + (f" naming {n} path(s)" if n else "")
                     + " (search log)")
    if seen:
        names = sorted(seen)
        facts.append("Paths retrieved: " + ", ".join(names[:12])
                     + (f" and {len(names) - 12} more" if len(names) > 12
                        else "")
                     + " (search log; their contents were not summarised)")
    sections = {
        "facts": facts,
        "searched_empty": [f"{_describe(t)} (search log)" for t in steps
                           if t.get("empty")],
        "open": [("Not answered: " + " ".join(question.split())[:200])
                 if question.strip() else "No conclusion was written."],
        "next": ["Read the retrieved paths above before relying on them."
                 if seen else "Search for the answer directly; deep thinking "
                              "retrieved nothing to read."],
    }
    lead = (f"[machine-built hand-off: deep thinking wrote none ({why[:200]})."
            f" It lists what was run and retrieved, and states no "
            f"conclusion.]")
    rendered, cut = _cut(_render(sections, lead, handle))
    return rendered, _stats(sections, 0, rendered, True, True, cut)


PLAN_MAX_ITEMS = {"files": 15, "order": 12, "decisions": 8, "risks": 6}
_PLAN_PATTERNS = {"files": r"files?(?:\s+to\s+(?:create|change|touch))?",
                  "order": r"order|steps?|plan",
                  "decisions": r"(?:key\s+)?decisions?",
                  "risks": r"risks?"}
_PLAN_HEADING = re.compile(
    r"^\s*(?:#+\s*)?\**\s*(" + "|".join(_PLAN_PATTERNS.values())
    + r")\s*\**\s*(?::\s*\**\s*(.*))?$", re.IGNORECASE)


def plan_handoff(text: str, trace: list, seen: set[str],
                 handle: str = "", root: str | None = None
                 ) -> tuple[str, dict]:
    """The planner's write-up as the fixed four-section plan, and its stats.
    Text before any heading is read as ORDER. A KEY DECISION is checked like
    a fact: its citation stands when this run retrieved it, else it crosses
    labelled. The other sections are the plan itself, not claims."""
    out: dict = {k: [] for k, _ in PLAN_SECTIONS}
    cur, structured, in_fence = "order", False, False
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if not in_fence:
            m = _PLAN_HEADING.match(line)
            if m:
                name = m.group(1).strip()
                cur = next(k for k, pat in _PLAN_PATTERNS.items()
                           if re.fullmatch(pat, name, re.IGNORECASE))
                structured = True
                rest = (m.group(2) or "").strip().strip("*").strip()
                if rest and not _NONE.match(rest):
                    out[cur].append(rest)
                continue
        if not line.strip():
            continue
        item = _BULLET.sub("", line, count=1).strip()
        if _NONE.match(item):
            continue
        if not _BULLET.match(line) and out[cur] and (in_fence or raw[:1] in
                                                    " \t"):
            out[cur][-1] += "\n  " + line.strip()
        else:
            out[cur].append(item)
    verified = 0
    labelled = []
    ev = _evidence_state(trace, root)
    for i, d in enumerate(out["decisions"]):
        t, ok = _label_fact(d, seen, ev if i < PLAN_MAX_ITEMS["decisions"]
                            else dict(ev, left=0))
        labelled.append(t)
        verified += ok
    out["decisions"] = labelled
    parts = []
    for key, title in PLAN_SECTIONS:
        items = out[key]
        cap = PLAN_MAX_ITEMS[key]
        lines = [f"- {i}" for i in items[:cap]] or ["- none"]
        if len(items) > cap:
            lines.append(f"- (+{len(items) - cap} more"
                         + (f"; trace handle {handle}" if handle else "") + ")")
        parts.append(title + "\n" + "\n".join(lines))
    rendered, cut = _cut("\n".join(parts), extra=ev["chars"])
    stats = {"facts": len(out["decisions"]), "verified": verified,
             "unverified": len(out["decisions"]) - verified,
             "searched_empty": sum(1 for t in trace
                                   if t.get("empty") and "hop" in t),
             "open": len(out["risks"]), "chars": len(rendered),
             "machine_built": False, "structured": structured, "cut": cut,
             "plan": {k: len(v) for k, v in out.items()},
             "excerpts": ev["excerpts"], "excerpt_chars": ev["chars"],
             "excerpts_skipped": ev["skipped"],
             "citations_removed": ev["removed"]}
    return rendered, stats


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
            "filling this conversation, then returns one short hand-off: "
            "facts with file citations, searches that found nothing, open "
            "questions and a next step. Use it when an answer needs searching you do not "
            "need to watch. Do not use it for a single lookup -- call the "
            "search tool yourself, it is faster. Its searches cover the "
            "library source the remote code-intelligence service holds; the "
            "user's own files are read with your client's own file tools."),
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
