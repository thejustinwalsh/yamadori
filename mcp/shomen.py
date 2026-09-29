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
    checked". At the turn cap, the helper-budget stop, or a hop that ends in
    its reasoning, the landing prompt (LANDING) requires the hand-off from
    what was gathered, written by the helper itself -- a tool result is never
    pasted across. If the helper writes nothing, `machine_handoff()` builds
    one from the trace (the source lines read that contain the question's
    words, re-read from the held source and labelled; the paths retrieved;
    the searches run), labelled as machine-built. An empty
    hand-off is impossible. Nothing cuts it: its length is bounded by the
    job's own generation budget (tiers.JOB_THINKING plus its answer
    allowance); MAX_FINDING_CHARS and its "[hand-off cut ...]" marker were
    removed on 2026-09-27 (operator: an unmeasured number, and the cut plan
    broke pagoda-h2).

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
four TRIGGERS decide it (mcp/deep.py) -- the model's own `yama_think_deeply` call,
struggle the proxy detects, a known-hard area, a new task's kickoff (the
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
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("YAMADORI_HELPER_MODEL", "bonsai")


def _model() -> str:
    """The second brain works on the conversation's model (MAX MODE, mcp/max_mode.py: a max conversation's jobs run on
    the max model); MODEL outside a request or with max mode off."""
    import max_mode
    return max_mode.current(MODEL)
# MAX_HOPS (16) was removed on 2026-09-22 at the operator's instruction: a
# count that ENDED the run hid a lying tool behind a slow failure. Since
# 2026-09-23 a per-run tool-turn limit (tiers.tool_turn_limit: 10, 20 at max)
# LANDS the run instead -- tools withdrawn, finding written -- and the trace
# records it. See _investigate().

# ---------------------------------------------------------------------------
# LAYA'S ROUTER IS GONE (2026-09-27, docs/CONSTANTS-AUDIT.md). route_in()
# and distil(), their Laya call (_laya, _choice_averaged), its gate and
# ordering count (MARGIN_GATE 0.3, LAYA_PERMUTATIONS 2) and the
# uncalibrated thresholds (T_INVESTIGATE, T_ANSWERED) had no production
# caller: deep thinking's triggers decide (mcp/deep.py), distil measured
# non-discriminating (n=29, 29/29 from_the_files; docs/FINDINGS.md #17),
# and 0.3 was never reproduced (bench/mechanisms/selectors.py). The
# deterministic citation check below is the authority on what was read.
# ---------------------------------------------------------------------------
# NO CAP ON WHAT CROSSES (operator, 2026-09-27: "You made that limit up,
# unfounded."). MAX_FINDING_CHARS (1,400, then 6,000 as a "breaker") cut a
# hand-off or a plan mid-sentence and wrote "[hand-off cut at N characters
# of M]" into it; pagoda-h2's 8,958-character kickoff plan was cut so, main
# echoed the marker ("[The rest of the second model's response continues
# below]") and stopped without a call. Neither number was measured. The
# bound is each job's own generation budget (tiers.JOB_THINKING and the
# answer allowance), and a result that does not fit main's window is the
# window check's (proxy.fit_window, check_client_prompt). The per-section
# item caps went with it: they dropped items behind a "(+N more)" line.

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
# still handed back, labelled as reasoning. No word target (2026-09-27,
# docs/CONSTANTS-AUDIT.md: the 250 was never measured), and no style block:
# its "may be read by a classifier" rules were for Laya's distil, removed on
# measurement (docs/FINDINGS.md #17).
# handoff/2 (operator, 2026-09-28, after pagoda-h6: "facts and the next
# concrete action, no doubt"): a point the sources did not settle is
# DECIDED in NEXT STEP; OPEN QUESTIONS holds a gap stated as a fact, never
# "what would settle it" (homework); NEXT STEP is one concrete action; and
# handoff() drops any line of homework, instability or version history
# (skill_limits.doubt), recorded as `doubt_dropped`.
HANDOFF_PROMPT_VERSION = "handoff/2"
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
its own line, starting with "- ".

FACTS
SEARCHED, FOUND NOTHING
OPEN QUESTIONS
NEXT STEP

Route each thing you know with this table:

| what you have | section | end the line with |
|---|---|---|
| a statement you read in a file | FACTS | the path and line, as src/file.ts:42 |
| a statement you read on a web page | FACTS | the page's URL, then {WEB_LABEL} |
| a statement from the knowledge base | FACTS | its id, as skill:3f2a9c0e1b4d |
| a statement you worked out without reading it | FACTS | {REASONING_LABEL} |
| a search that returned nothing useful | SEARCHED, FOUND NOTHING | the tool and what you searched for |
| a point the sources did not settle | NEXT STEP | the choice you make for it, stated as the action |
| something the index does not hold | OPEN QUESTIONS | the gap, stated as a fact, like "the index holds no koota source" |
| what the engineer does next | NEXT STEP | one concrete action: the file to write, the change to make or the command to run |

Write "- none" under a section that has nothing. Write what the code DOES,
not what you did to find it. The engineer acts on the hand-off as written,
so every line is a fact, a gap stated as a fact, or an action.

When a mockup, diagram or design sketch would help the answer, draw it with
yama_generate_image. Then look at it with yama_describe_image, passing the url it
returned and a question such as "Does this show <what you asked for>?". If
the picture is off, adjust the prompt and draw it again. Put the markdown
line of the final image under FACTS: the link is the only part of the image
that crosses back. Say in words what the final image shows."""

# What a landing asks for -- the tool-turn cap and the helper-budget stop
# both. It REQUIRES the hand-off from what was gathered: the tool results are
# already in this context, and turning them into FACTS is the helper's job.
# The proxy never pastes raw tool output across.
LANDING = ("Stop searching. Write the hand-off now from what you have "
           "gathered, in the four sections FACTS, SEARCHED, FOUND NOTHING, "
           "OPEN QUESTIONS and NEXT STEP. Turn what the searches returned "
           "into FACTS in your own words, each ending with its path:line. "
           "State a gap as a fact under OPEN QUESTIONS, and the next "
           "concrete action under NEXT STEP.")

# THE PLAN JOB (Phase 0.6, trigger 4: TASK KICKOFF; operator, 2026-09-24).
# Every new task sends its PLANNING here, whatever its length (operator,
# 2026-09-27): the Octopus pilot's V0 steps 1-3 spent 12-14k reasoning tokens
# each planning on main (#18, docs/SELF-IMPROVEMENT-LOG.md), in the context
# every later step re-reads. The second brain writes a compact plan in four
# fixed sections; the proxy returns it as the yama_plan tool result
# (2026-09-27) and main starts acting. Same shape as the hand-off: a routing table, no prohibition, no
# word target (PLAN_TARGET_WORDS, 300, never measured; removed 2026-09-27
# with v1's "Aim for under N words"). REMOVED 2026-09-27 (operator:
# task-targeted steering in prompts; skills are the channel): the confirm-working-directory
# step and relative-paths rule (#32) and the browser-app entry-first row
# (#55), their enforcement in plan_handoff and switch plan_entry_first.
# THE PLAN STATES DECISIONS (operator, 2026-09-28, after pagoda-h6: "plans
# don't sow doubt, I have been building with all of those APIs in alpha
# state with none of those concerns"; "This model overthinks, don't give it
# reason to!"). pagoda-h6's plan carried RISKS written as homework -- "Check
# @react-three/fiber README for the exact prop API before writing main.tsx",
# "Verify meshStandardMaterial props at build time", "0.186 may shift API"
# -- and main probed node_modules for hours. The cause was this prompt's own
# row: `| something likely to go wrong | RISKS | the risk and how to check
# for it |`. plan/3: RISKS is gone; CONSTRAINTS holds the properties the
# code must have, stated as facts ("petals share one InstancedMesh"); an
# unknown is DECIDED in KEY DECISIONS; ORDER's last step runs the result.
# plan_handoff drops any line that still assigns verification homework,
# warns of instability or recites version history (skill_limits.doubt),
# recorded as `doubt_dropped`. A RISKS heading the model writes anyway is
# read as CONSTRAINTS and filtered the same way.
PLAN_PROMPT_VERSION = "plan/4"
PLAN_SECTIONS = (("files", "FILES"),
                 ("order", "ORDER"),
                 ("decisions", "KEY DECISIONS"),
                 ("constraints", "CONSTRAINTS"))
_PLAN_TABLE = """| what you have | section | how to write it |
|---|---|---|
| a file to create or change | FILES | the path, then what it holds, in a few words |
| a task that asks for the code or the answer in the reply ("just the code", "show me", a question) | FILES | "- none: the deliverable is the reply"; ORDER ends with writing the reply |
| a step | ORDER | one action per line, first step first; the last step runs the result (the build, the tests or the page) |
| a choice the task leaves open (library, API, structure, data format) | KEY DECISIONS | the choice, stated as decided, and the reason; the source when you read one (path:line, or the URL then {web}) |
| a point you are unsure of | KEY DECISIONS | the choice you make for it, stated as decided |
| a property the code must hold | CONSTRAINTS | the property, stated as a fact, like "petals share one InstancedMesh" |

Write "- none" under a section that has nothing. The engineer acts on every
line as written, so each line is a decision, a step or a fact. Keep names,
paths, versions and commands verbatim. Plain engineering English: one idea
per line, the noun named, active voice.""".replace("{web}", WEB_LABEL)
PLAN_SYSTEM = f"""You are planning a software task for another engineer, who
will carry it out with their own tools. You have search tools for library
source, skills and the web. The engineer gets your plan and nothing else:
not your searches, not your reasoning.

Work in this order:
1. Read the task. Search only for what a decision depends on: an API, a
   version, a library the task names.
2. Write the plan.

THE PLAN has exactly these four sections, in this order. Put each item on its
own line, starting with "- ".

FILES
ORDER
KEY DECISIONS
CONSTRAINTS

Route each thing with this table:

{_PLAN_TABLE}"""

# PLAN_SYSTEM_V2 (#60 in docs/SELF-IMPROVEMENT-LOG.md; switch plan_prompt,
# YAMADORI_PLAN_PROMPT_V2=0 or X-Yamadori-Features {"plan_prompt": false}
# gives PLAN_SYSTEM back). The v0f-V0-xhigh-1 kickoff plan's reasoning
# (90,745 characters over three hops) shows two things the prompt caused:
# - "The engineer gets your plan and nothing else" read as "the engineer
#   has no task text": 8 re-readings, and paragraphs deciding whether to
#   copy the task's constants into FILES. Main DOES have the task (the plan
#   is prefilled into the conversation that carries it), so V2 says so.
# - "Aim for under 300 words": 85 word-count passages and 10-12 drafts of
#   the four sections (a complete draft existed ~5,100 tokens into hop 1).
#   V2 drops the number; its per-item style sentences ("One short line per
#   item ... referred to, not copied", from that one run) went on 2026-09-27
#   (docs/CONSTANTS-AUDIT.md). Nothing caps what crosses (NO CAP ON WHAT
#   CROSSES, 2026-09-27).
# And the tools line matches what is offered: "this project's docs" have
# not been readable by any model since 2026-09-25, and a plan offered no
# tools is told so. UNMEASURED WORDING. Both prompts share _PLAN_TABLE
# (plan/3, above).
_PLAN_V2_TOOLS = ("You have search tools for held library source, the "
                  "knowledge base and the web.")
_PLAN_V2_NO_TOOLS = ("Plan from the task and what you know; there is "
                     "nothing to search for this task.")
_PLAN_V2_STEP_TOOLS = ("1. Read the task. Search only for what a decision "
                       "depends on: an API, a version, a library the task "
                       "names.")
_PLAN_V2_STEP_NO_TOOLS = "1. Read the task."
PLAN_SYSTEM_V2 = f"""You are planning a software task for another engineer, who
will carry it out with their own tools. The engineer has the task text; your
plan gives it an order and settles every decision it leaves open. They do not
see your searches or your reasoning. @@TOOLS@@

Work in this order:
@@STEP1@@
2. Write the plan once, in the four sections below.

THE PLAN has exactly these four sections, in this order. Put each item on its
own line, starting with "- ".

FILES
ORDER
KEY DECISIONS
CONSTRAINTS

Route each thing with this table:

{_PLAN_TABLE}"""


def _plan_switch(name: str, flag: bool | None = None) -> bool:
    """A plan switch (tiers.PLAN_SWITCHES): the caller's value, else the
    environment's (tiers.BEHAVIOURS)."""
    if flag is not None:
        return bool(flag)
    import tiers
    return tiers.behaviour(None, name)


def plan_system(tools: bool = True, v2: bool | None = None) -> str:
    """The plan job's system prompt: PLAN_SYSTEM_V2 when the plan_prompt
    switch is on (its tools line by whether `tools` are offered), else
    PLAN_SYSTEM."""
    if not _plan_switch("plan_prompt", v2):
        return PLAN_SYSTEM
    return (PLAN_SYSTEM_V2
            .replace("@@TOOLS@@", _PLAN_V2_TOOLS if tools
                     else _PLAN_V2_NO_TOOLS)
            .replace("@@STEP1@@", _PLAN_V2_STEP_TOOLS if tools
                     else _PLAN_V2_STEP_NO_TOOLS))


# THE PLAN'S TOOLS. The plan gets our tools, as investigate does, with one
# rule: run_check runs a project's check by label, so it needs a bound
# repository, and without one it is left out (v0f-V0-xhigh-1: the planner
# ran run_check with no repository). REMOVED 2026-09-27
# (docs/CONSTANTS-AUDIT.md: chosen from one v0f run, n=1; off since that
# morning): #60's gate that offered the tools only when the task named held
# source (PLAN_TOOL_SITUATIONS, switch plan_tools) and the plan's own budget
# (tiers.PLAN_TOOL_TURNS 2 and PLAN_SECONDS 150, switch plan_budget). The
# plan lands at the tier's tool-turn limit like investigate.
_ROOT_ONLY_TOOLS = {"run_check"}


def plan_tools(tools: list[dict], source_root: str | None = None
               ) -> tuple[list[dict], dict]:
    """(the tools the plan job gets, the record of why): every tool, less
    run_check when no repository is bound."""
    if source_root:
        return tools, {"offered": len(tools), "why": "repository bound"}
    kept = [t for t in tools if (t.get("function") or {}).get("name")
            not in _ROOT_ONLY_TOOLS]
    return kept, {"offered": len(kept),
                  "why": "no repository bound: run_check left out"
                  if len(kept) < len(tools) else "every tool"}


PLAN_LANDING = ("Stop searching. Write the plan now from what you have, in "
                "the four sections FILES, ORDER, KEY DECISIONS and "
                "CONSTRAINTS, each decision stated as decided.")

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
# the web (mcp/research_tools.py: read_web_page, search_web) and the
# service's knowledge base (find_in_knowledge_base: the armed skills). A web
# fact cites its URL and is labelled "(web)"; a knowledge-base fact cites
# skill:<id> (the store's 12-hex id). All are checked
# against what the run retrieved, exactly as a path is. Since 2026-09-25 the
# knowledge base holds no developer docs, so it can no longer hand back a
# docs/FILE.md:N citation; `md` stays in _PATH for the READMEs a held
# package's index carries (read_file_range).
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
    # Each job's hops think at most tiers.JOB_THINKING[job] (operator,
    # 2026-09-25): the job is the one run() is executing on this thread.
    job = getattr(_CURRENT, "job", None)
    shaped = model.shape(payload, effort=payload.get("_effort_tier") or "max",
                         role="helper",
                         step_cap=tiers.JOB_THINKING.get(job),
                         # a research hop's nudge names its action: the
                         # search, or the hand-off (tiers.HELPER_NUDGE_MESSAGE)
                         nudge=tiers.helper_nudge(job))
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


def _helper_budget() -> int:
    """The helper's slice of the unified KV pool, from budget.py.

    Read rather than hardcoded: the pool follows `-c` in config.yaml, and a
    second copy of the number here would be wrong the first time that changes.
    """
    try:
        import budget
        return int(budget.budgets().get("helper") or 49152)
    except Exception:                                            # noqa: BLE001
        return 61440


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


# THE RESEARCH JOBS' SEED LINE (#52 in docs/SELF-IMPROVEMENT-LOG.md,
# remedy 7 of docs/research/OVERTHINKING.md; 2026-09-26). In 4 of the 6
# struggle runs of Octopus v0e p2 the second brain read the word as data
# about the task: "'irres' is an anagram", "the inspiration word in these
# benchmarks is usually an anagram of a key word from the answer", "'encab'
# -- probably encode/cab... encoding, base64 encoding?" (session.p2.jsonl;
# that the encab hand-off led to the base64 data-URL attempts, #46, is
# inferred). An investigation or a plan READS a task for clues, so for those
# two jobs the line first says where the word comes from -- drawn at random,
# independent of the task -- and then gives it; the one prohibition names the
# observed failure (a clue, an encoding), the other the #13 one (not in the
# output). The fix-up and fan-out jobs keep SEED_LINE. The word is the
# ledger's (proxy.ledger_seed), so a replay renders the same bytes. The
# ANSWER's phrase, PHRASES["seed"], is unchanged. UNMEASURED WORDING. Off:
# YAMADORI_SEED_FRAME=0 or X-Yamadori-Features {"seed_frame": false}.
SEED_LINE = ("\n\nInspiration word: {word} -- let it shape how you approach "
             "the task; it is not part of the answer, so it does not appear "
             "in your code or text.")
SEED_LINE_RESEARCH = ("\n\nFor variety, a word drawn at random from the "
                      "vocabulary, independent of this task and of anything "
                      "in it: {word}. It is not a clue, a code or an anagram "
                      "of anything here; let it shape only how you approach "
                      "the work, and leave it out of what you write.")
RESEARCH_JOBS = ("investigate", "plan")


def seed_frame_on(flag: bool | None = None) -> bool:
    """The research seed line's switch: the request's (a header), else
    YAMADORI_SEED_FRAME (default on)."""
    if flag is not None:
        return bool(flag)
    import tiers
    return tiers._env_on("YAMADORI_SEED_FRAME")


def seed_phrase(word: str, where: str, job: str | None = None,
                frame: bool | None = None) -> str:
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
    if job in RESEARCH_JOBS and seed_frame_on(frame):
        return SEED_LINE_RESEARCH.format(word=word)
    return SEED_LINE.format(word=word)


def opening(seeds) -> str:
    """The investigate fold-back's visible opening, prefilled as main's
    content. Ends on the measured-safe "After thinking deeply,"."""
    line = seed_line(seeds)
    return (line + " " if line else "") + PHRASES["investigate"]


# The job run() is executing on this thread; _post reads it for the job's
# thinking cap (tiers.JOB_THINKING).
_CURRENT = __import__("threading").local()


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
        prev, _CURRENT.job = getattr(_CURRENT, "job", None), job
        try:
            res = _run_job(job, spec)
        finally:
            _CURRENT.job = prev
    res["job"] = job
    return res


def _run_job(job: str, spec: dict) -> dict:
    if job in ("investigate", "plan"):
        res = investigate(spec["question"], spec["tools"],
                          spec["run_tool"], spec.get("context", ""),
                          on_think=spec.get("on_think"),
                          tier=spec.get("tier", "max"),
                          effort=spec.get("effort"),
                          seed=spec.get("seed"), mode=job,
                          seed_frame=spec.get("seed_frame"),
                          # #60: tiers.plan_switches(tier) from the caller;
                          # absent, each switch reads the environment.
                          plan_switches=spec.get("plan_switches"),
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
    for e in errors or []:
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
    head = (f"The request:\n{(request or '').strip() or '(none)'}\n\n")
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


def fix_rejection(original: str, code: str | None, errors: list,
                  why: str | None) -> str | None:
    """Why a fix-up's last version must NOT replace the model's code, or
    None when it may. The rule (pre-deploy review, 2026-09-24): the version
    came from a CLOSED fence (`_largest_block`), the reply was not cut off
    (finish_reason "length"; both surface as `why`), and it parses with no
    errors left. REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md): the length
    floor FIXUP_MIN_KEEP (half the original's non-blank characters), a
    CHOICE with no fix-up length distribution behind it. `original` stays
    in the signature for the callers."""
    if why:
        return why
    if code is None:
        return "no version came back"
    if errors:
        return (f"{len(errors)} error{'s' if len(errors) != 1 else ''} "
                f"still in the repaired version")
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
            payload = {"model": _model(), "messages": convo, "tools": [],
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
                source_root: str | None = None,
                seed_frame: bool | None = None,
                plan_switches: dict | None = None) -> dict:
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
    run_rec: dict = {}
    res = _investigate(question, tools, run_tool, context, hops, on_think,
                       seed, tier=tier, effort=effort, mode=mode,
                       source_root=source_root, seed_frame=seed_frame,
                       plan_switches=plan_switches,
                       run_rec=run_rec)
    # Per generation: seconds, prompt and completion tokens, finish (#60: the
    # v0f plan's breakdown had to be inferred from the llama-swap log).
    res["generations"] = run_rec.get("generations") or []
    if mode == "plan":
        st = res.get("handoff_stats")
        if isinstance(st, dict):
            st["plan"] = dict(st.get("plan") or {},
                              run={k: run_rec.get(k) for k in (
                                  "switches", "tools", "thinking_cap",
                                  "landed", "generations")})
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
                 source_root: str | None = None,
                 seed_frame: bool | None = None,
                 plan_switches: dict | None = None,
                 run_rec: dict | None = None) -> dict:
    started = time.time()
    plan = mode == "plan"
    landing = PLAN_LANDING if plan else LANDING
    import tiers
    run_rec = run_rec if run_rec is not None else {}
    gens: list[dict] = run_rec.setdefault("generations", [])
    psw: dict = {}
    if plan:
        # #60: the plan's prompt behind its switch (plan_prompt); its tools
        # are investigate's, less run_check with no repository.
        psw = {"plan_prompt": _plan_switch(
            "plan_prompt", (plan_switches or {}).get("plan_prompt"))}
        tools, tools_rec = plan_tools(tools, source_root)
        run_rec.update(switches=psw, tools=tools_rec,
                       thinking_cap=tiers.JOB_THINKING.get("plan"))
    convo = [{"role": "system", "content": plan_system(
        tools=bool(tools), v2=psw.get("plan_prompt"))
              if plan else SYSTEM}]
    if context:
        convo.append({"role": "user",
                      "content": f"Context from the conversation:\n{context}"})
    convo.append({"role": "user", "content": question + (
        seed_phrase(seed["word"], where="shomen", job=mode,
                    frame=seed_frame) if seed else "")})

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
    limit = tiers.tool_turn_limit(tier)
    run_rec.setdefault("landed", None)

    def _gen(p: dict) -> dict:
        t0 = time.time()
        d = _post("/v1/chat/completions", p)
        u = d.get("usage") or {}
        gens.append({"seconds": round(time.time() - t0, 1),
                     "prompt": u.get("prompt_tokens"),
                     "completion": u.get("completion_tokens"),
                     "finish": ((d.get("choices") or [{}])[0]
                                .get("finish_reason")),
                     "tools": len(p.get("tools") or [])})
        return d
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
                run_rec["landed"] = "turn cap"
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
        payload = {"model": _model(), "messages": convo,
                   "tools": [] if last else tools,
                   "max_tokens": 1500,
                   "_effort_tier": tier, "_effort_override": effort}
        try:
            d = _gen(payload)
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
                d = _gen(dict(payload, tools=[], messages=convo))
                spent += int((d.get("usage") or {}).get("total_tokens") or 0)
                msg = d["choices"][0]["message"]
            except Exception as e:                               # noqa: BLE001
                return _fail(f"{type(e).__name__}: {e}", trace, started, spent,
                             question, seen_paths)
            return _finish((msg.get("content") or "").strip(), trace,
                           seen_paths, started, spent, question, mode,
                           source_root, context, seed=seed,
                           seed_frame=seed_frame)

        calls = msg.get("tool_calls") or []

        if not calls:
            finding = (msg.get("content") or "").strip()
            if not finding and msg.get("reasoning_content"):
                # A HOP THAT ENDED IN ITS REASONING: no tool call, no text.
                # Until 2026-09-26 this failed at once as "ran out of budget
                # while reasoning", whatever ended it -- and the deploy check
                # that day (handle d28941fb: 15 searches, 204 s, ContextNode.js
                # and UniformNode.js read) handed main a search log with no
                # conclusion, although no budget was near its end (the
                # helper's window ~15-20k of 49,152, 15 of 20 tool turns).
                # Now the hop's end is RECORDED (finish_reason, sizes, a tool
                # call left inside the reasoning), and the run LANDS once --
                # tools withdrawn, LANDING asks for the hand-off from what was
                # gathered -- as the turn cap and the KV stop already did.
                ended = _ended_in_reasoning(d, msg, hop)
                trace.append({"tool": "(no call)", "args": {},
                              "result": ended})
                if last:
                    return _fail(ended + " (on the landing itself)", trace,
                                 started, spent, question, seen_paths,
                                 root=source_root)
                convo.append({"role": "user", "content": landing})
                try:
                    d = _gen(dict(payload, tools=[], messages=convo))
                    spent += int((d.get("usage") or {}).get("total_tokens")
                                 or 0)
                    msg = d["choices"][0]["message"]
                except Exception as e:                           # noqa: BLE001
                    return _fail(f"{ended}; the landing raised "
                                 f"{type(e).__name__}: {e}", trace, started,
                                 spent, question, seen_paths,
                                 root=source_root)
                if on_think and msg.get("reasoning_content"):
                    try:
                        on_think(msg["reasoning_content"])
                    except Exception:                            # noqa: BLE001
                        pass
                finding = (msg.get("content") or "").strip()
                if not finding:
                    return _fail(
                        f"{ended}; the landing (tools withdrawn) wrote no "
                        f"text either: "
                        f"{_ended_in_reasoning(d, msg, None)}", trace,
                        started, spent, question, seen_paths,
                        root=source_root)
            return _finish(finding, trace, seen_paths, started, spent,
                           question, mode, source_root, context, seed=seed,
                           seed_frame=seed_frame)
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
            # Same breaker as the proxy's loops, with a marker and the next
            # range to request -- a silent cut handed this context half a file
            # under a header promising all of it (docs/CONSTRAINTS.md 12).
            shown = repeats.cap_tool_result(out, fn, args)
            key = repeats.normalise(fn, args if isinstance(args, dict) else {})
            # `empty` and `paths` feed the hand-off: an empty search is listed
            # under SEARCHED, FOUND NOTHING so the main model does not repeat
            # it, and a machine-built hand-off reports hit counts from them.
            trace.append({"hop": hop, "tool": fn, "args": args,
                          "chars": len(out), "empty": repeats._empty(out),
                          "paths": len(found),
                          # The package versions it read: the hand-off's
                          # evidence is read from these (resolve_source).
                          "versions": versions_in(out),
                          # How often it had already made this same call
                          # (repeats.normalise) in this run.
                          "repeat": sum(1 for t in trace
                                        if t.get("key") == key),
                          "key": key,
                          "ms": round((time.time() - t0) * 1000)})
            convo.append({"role": "tool", "tool_call_id": c.get("id"),
                          "content": shown})

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
            source_root: str | None = None, context: str = "",
            seed: dict | None = None, seed_frame: bool | None = None
            ) -> dict:
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
    # A plan's claims are its KEY DECISIONS' citations (#60): FILES names the
    # files to CREATE, and counting them made v0f's plan "cited 0,
    # unsupported 9" -- nine paths that do not exist yet, not nine claims.
    claimed = _cited(_plan_claims(finding) if mode == "plan" else finding)
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
                                      "it returned no text", handle,
                                      root=source_root)
    # Which seed line the job's user message carried (#52).
    stats["seed_frame"] = ("research" if seed and seed_frame_on(seed_frame)
                           else "plain" if seed else None)
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
    # `distil()`, kept for a while as the reference implementation of the
    # call pattern, was deleted on 2026-09-27 (no caller). See docs/LAYA.md
    # Part II.
    return out


_CALL_IN_REASONING = re.compile(r"<tool_call>|<function=[\w.-]+>")


def _ended_in_reasoning(d: dict, msg: dict, hop: int | None) -> str:
    """How a generation that wrote neither a tool call nor text ended, in
    one line: which hop, its finish_reason and sizes, and whether its
    reasoning holds a tool call in the template's markup (the server parses
    a call only after the reasoning closes). It says "token limit" only when
    the server said `length`."""
    ch = (d.get("choices") or [{}])[0]
    fr = ch.get("finish_reason") or "unreported"
    rc = msg.get("reasoning_content") or ""
    toks = (d.get("usage") or {}).get("completion_tokens")
    who = f"hop {hop + 1}" if hop is not None else "it"
    how = ("hit its token limit in its reasoning" if fr == "length"
           else "ended in its reasoning")
    return (f"{who} {how}: no tool call, no text (finish_reason {fr}, "
            f"{len(rc)} characters of reasoning"
            + (f", {toks} tokens" if toks else "")
            + ("; a tool call inside the reasoning, unparsed"
               if _CALL_IN_REASONING.search(rc) else "") + ")")


def _fail(why: str, trace: list, started: float, spent: int,
          question: str = "", seen: set[str] | None = None,
          root: str | None = None) -> dict:
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
                                  f"investigation failed: {why}", handle,
                                  root=root)
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
# Every item crosses (2026-09-27): the per-section item caps (facts 12,
# searched-empty 8, open 5, next 3; CHOICES, never measured) dropped the
# rest behind a "(+N more; trace handle ...)" line main could not follow.
# ---------------------------------------------------------------------------

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
    ev.setdefault("labels", []).append(label)
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
    """The sections as they cross: every item (NO CAP ON WHAT CROSSES)."""
    parts = [lead] if lead else []
    for key, title in SECTIONS:
        items = sections.get(key) or []
        lines = [f"- {i}" for i in items] or ["- none"]
        parts.append(title + "\n" + "\n".join(lines))
    return "\n".join(parts)


def _stats(sections: dict, verified: int, text: str, machine: bool,
           structured: bool, ev: dict | None = None) -> dict:
    facts = len(sections.get("facts") or [])
    out = {"facts": facts, "verified": verified,
           "unverified": facts - verified if not machine else 0,
           "searched_empty": len(sections.get("searched_empty") or []),
           "open": len(sections.get("open") or []),
           "chars": len(text), "machine_built": machine,
           "structured": structured}
    if ev is not None:
        # THE EVIDENCE: excerpts inlined, their characters, verified facts
        # whose excerpt the budget left out, and citations removed because
        # the verifier could not read them.
        out.update(excerpts=ev["excerpts"], excerpt_chars=ev["chars"],
                   excerpts_skipped=ev["skipped"],
                   citations_removed=ev["removed"],
                   # "three@0.185.1 src/nodes/core/ContextNode.js:267-275":
                   # what the proxy's ALREADY_THOUGHT return names.
                   excerpt_labels=list(ev.get("labels") or []))
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
    # Every fact crosses; the excerpt budget (EXCERPT_TOTAL_CHARS) goes to
    # the first ones.
    for f in sections["facts"]:
        labelled, ok = _label_fact(f, seen, ev)
        facts.append(labelled)
        verified += ok
    sections["facts"] = facts
    # OPEN QUESTIONS and NEXT STEP are not checked: they name what is NOT
    # known, and a NEXT STEP may name a file of the user's own project for
    # main to read with the harness's tools (Phase 0.5). Every section but
    # the search log passes the ASSURED VOICE filter (handoff/2, operator
    # 2026-09-28): homework, instability and history do not cross.
    doubt_dropped: list[dict] = []
    for key in ("facts", "open", "next"):
        sections[key] = _assured(sections[key], key, doubt_dropped)
    have = {s.lower() for s in sections["searched_empty"]}
    for t in trace:
        if t.get("empty") and "hop" in t:
            d = _describe(t)
            if not any(d.lower() in h or h in d.lower() for h in have):
                sections["searched_empty"].append(d + " (search log)")
                have.add(d.lower())
    rendered = _render(sections, handle=handle)
    stats = _stats(sections, verified, rendered, False, structured, ev)
    stats["prompt"] = HANDOFF_PROMPT_VERSION
    stats["doubt_dropped"] = doubt_dropped
    return rendered, stats


def machine_handoff(question: str, trace: list, seen: set[str], why: str,
                    handle: str = "", root: str | None = None
                    ) -> tuple[str, dict]:
    """The hand-off the PROXY builds when the helper wrote none.

    An empty hand-off is impossible (operator, 2026-09-23): whatever the
    investigation ran and retrieved crosses back, labelled as machine-built,
    as log lines -- the paths retrieved, then one search-log line per
    distinct call (repeats counted) with what it returned. It states no
    conclusion, because none was written.

    REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md; operator: "You made that
    limit up, unfounded."): MACHINE-BUILT EVIDENCE (2026-09-26), the source
    lines picked by the question's words (MACHINE_EXCERPTS 4, 400 hits a
    step, two terms, code terms x3, idf weights, a stoplist -- all CHOICES
    from one deploy check, handle d28941fb, n=1), and the cuts paths[:12],
    asked[:200] and why[:360]. Every path, the whole question and the whole
    reason cross.
    """
    steps = [t for t in trace if "hop" in t]
    facts: list[str] = []
    if seen:
        facts.append("Paths retrieved: " + ", ".join(sorted(seen))
                     + " (search log; their contents were not summarised)")
    facts += _search_log([t for t in steps if not t.get("empty")])
    asked = _asked(question)
    if seen:
        nxt = "Go on from what is known and act on it."
    else:
        nxt = ("Search for the answer directly; deep thinking retrieved "
               "nothing to read.")
    sections = {
        "facts": facts,
        "searched_empty": _search_log([t for t in steps if t.get("empty")],
                                      counts=False),
        "open": [("Not answered: " + asked)
                 if asked else "No conclusion was written."],
        "next": [nxt],
    }
    lead = (f"[machine-built hand-off: deep thinking wrote none ({why}). It "
            "lists what was run and retrieved, and states no conclusion.]")
    rendered = _render(sections, lead, handle)
    # The evidence record's counts (all zero: nothing is excerpted), so a
    # machine-built hand-off's stats have the written one's shape.
    return rendered, _stats(sections, 0, rendered, True, True,
                            _evidence_state(trace, root))


_QUESTION_MARK = re.compile(r"(?m)^\s*QUESTION:\s*")


def _asked(question: str) -> str:
    """The question itself, one line: the text after a trigger's last
    "QUESTION:" (deep._route_question wraps the user's words in a preamble,
    which OPEN QUESTIONS does not repeat), else all of it."""
    q = question or ""
    marks = list(_QUESTION_MARK.finditer(q))
    if marks:
        q = q[marks[-1].end():]
    return " ".join(q.split())


def _search_log(steps: list, counts: bool = True) -> list[str]:
    """One "(search log)" line per distinct call, in first-run order, with
    how many times it ran and what it returned."""
    seen: dict[str, list] = {}
    for t in steps:
        seen.setdefault(_describe(t), []).append(t)
    out = []
    for d, ts in seen.items():
        if not counts:
            out.append(d + (f", {len(ts)} times" if len(ts) > 1 else "")
                       + " (search log)")
            continue
        ch = sorted({int(t.get("chars") or 0) for t in ts})
        ps = sorted({int(t.get("paths") or 0) for t in ts})
        span = (lambda v: str(v[0]) if len(v) == 1 else f"{v[0]}-{v[-1]}")
        out.append(f"{d} returned {span(ch)} characters"
                   + (f" naming {span(ps)} path(s)" if ps[-1] else "")
                   + (f", {len(ts)} times" if len(ts) > 1 else "")
                   + " (search log)")
    return out


_PLAN_PATTERNS = {"files": r"files?(?:\s+to\s+(?:create|change|touch))?",
                  "order": r"order|steps?|plan",
                  "decisions": r"(?:key\s+)?decisions?",
                  # plan/3: a RISKS heading (plan/2's, or written anyway)
                  # is read as CONSTRAINTS and filtered like every line.
                  "constraints": r"(?:design\s+)?constraints?|risks?"}
_PLAN_HEADING = re.compile(
    r"^\s*(?:#+\s*)?\**\s*(" + "|".join(_PLAN_PATTERNS.values())
    + r")\s*\**\s*(?::\s*\**\s*(.*))?$", re.IGNORECASE)


def _plan_claims(text: str) -> str:
    """The KEY DECISIONS lines of a written plan: the only section whose
    items cite (plan_handoff checks them like facts). No headings at all:
    the whole text (an unstructured plan is checked as before)."""
    cur, out, headed = None, [], False
    for line in (text or "").splitlines():
        m = _PLAN_HEADING.match(line.rstrip())
        if m:
            headed = True
            cur = next(k for k, pat in _PLAN_PATTERNS.items()
                       if re.fullmatch(pat, m.group(1).strip(), re.I))
            if cur == "decisions" and m.group(2):
                out.append(m.group(2))
            continue
        if cur == "decisions":
            out.append(line)
    return "\n".join(out) if headed else (text or "")


def _assured(items: list[str], section: str, dropped: list[dict]
             ) -> list[str]:
    """ASSURED VOICE (skill_limits.doubt; operator, 2026-09-28): the items
    that state a fact, a decision or an action. A line of verification
    homework, instability or version history is dropped and recorded."""
    import skill_limits
    kept = []
    for it in items:
        why = skill_limits.doubt(it.split("\n", 1)[0])
        if why:
            dropped.append({"section": section, "line": it[:300],
                            "why": why})
        else:
            kept.append(it)
    return kept


_PLAN_UNCHECKED = re.compile(
    r"\s*(?:\(reasoning,? not checked against source\)|\(reasoning\)"
    r"|\[cites [^\]]*: unverified\])", re.IGNORECASE)


def _plan_decision(d: str, seen: set[str], ev: dict) -> tuple[str, bool]:
    """A KEY DECISION as it crosses: decided (plan/3). Its citation is
    checked like a fact's, so a verified one carries the lines it cites; an
    unreadable citation is removed; and nothing is labelled unchecked -- a
    decision is the planner's to make, and a label that says "not checked"
    reads as a reason to go and check (pagoda-h6)."""
    t, ok = _label_fact(d, seen, ev)
    return _PLAN_UNCHECKED.sub("", t).rstrip(), ok


def plan_handoff(text: str, trace: list, seen: set[str],
                 handle: str = "", root: str | None = None
                 ) -> tuple[str, dict]:
    """The planner's write-up as the fixed four-section plan, and its stats.
    Text before any heading is read as ORDER. A KEY DECISION's citation is
    checked like a fact's (_plan_decision); the other sections are the plan
    itself, not claims. Every line passes the ASSURED VOICE filter
    (_assured): homework, instability and history are dropped, recorded."""
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
    doubt_dropped: list[dict] = []
    for key, _title in PLAN_SECTIONS:
        out[key] = _assured(out[key], key, doubt_dropped)
    verified = 0
    labelled = []
    ev = _evidence_state(trace, root)
    for d in out["decisions"]:
        t, ok = _plan_decision(d, seen, ev)
        labelled.append(t)
        verified += ok
    out["decisions"] = labelled
    parts = []
    # Every item crosses (NO CAP ON WHAT CROSSES): a plan is kept compact
    # by its prompt, never trimmed here.
    for key, title in PLAN_SECTIONS:
        lines = [f"- {i}" for i in out[key]] or ["- none"]
        parts.append(title + "\n" + "\n".join(lines))
    rendered = "\n".join(parts)
    stats = {"facts": len(out["decisions"]), "verified": verified,
             "unverified": len(out["decisions"]) - verified,
             "searched_empty": sum(1 for t in trace
                                   if t.get("empty") and "hop" in t),
             "open": 0, "chars": len(rendered),
             "machine_built": False, "structured": structured,
             "prompt": PLAN_PROMPT_VERSION,
             "plan": {k: len(v) for k, v in out.items()},
             "doubt_dropped": doubt_dropped,
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
