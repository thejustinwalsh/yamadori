#!/usr/bin/env python
"""Every size limit on skills, in one place.

THESE ARE CHOICES, NOT MEASUREMENTS.

Nothing in this repo has measured how long a skill should be for this model.
The numbers follow published practice and the operator's direction:

  - Anthropic Agent Skills: a SKILL.md `name` is at most 64 characters and
    its `description` at most 1,024; the body is kept short and detail is
    disclosed progressively (docs/DECISION-TREES-AND-SKILLS.md §2.2,
    [R] S23-S25). Hermes enforces the same two caps
    (hermes-agent tools/skills_tool_plugin.py MAX_NAME_LENGTH /
    MAX_DESCRIPTION_LENGTH) and lints `name` to [a-z0-9_-].
  - SkillsBench v4 reports compact skills ahead of detailed ones and long
    documentation near zero ([R] S34, §2.3); instruction-following falls as
    instructions pile up ([R] S35).
  - Operator, 2026-09-26: "Our skills will be well tagged, categorized and
    compact because we have less context." So a skill is ATOMIC -- one
    trigger condition, a handful of moves -- and the per-skill caps below
    are tighter than the 2026-09-24 targets (aim 300-500 tokens, hard 800).

WHY THESE NUMBERS FOR THIS CONTEXT

An injected skill is appended to a user turn or a tool result and REPLAYED
byte for byte on every later request of the conversation (the ledger), so
its tokens are paid for the whole conversation, not once. THE BUDGET LIVES
IN THE SKILLS (operator, 2026-09-27: "I don't think a token budget on skill
selection is useful, I think the input skills need budgeted"): each skill is
capped where it is authored and ingested -- a body of aim 100-300 tokens,
hard 450, at most 6 items and 2 prohibitions -- and selection has no
per-turn token budget and no small count cap. It gives every area the user
ASKS for a slot, then the areas that implies, then strongly supported extras
(skill_select, THE SLOTS), under a sanity CEILING of 8 skills per decision
(a CHOICE, against a pathological request, not a budget). Eight bodies at
the hard cap are ~3,600 tokens, ~2.7% of main's 132,096-token window
(mcp/budget.py at -c 181248); at the store's median (~293 tokens, 2026-09-27
audit) ~2,300, ~1.8%.

PER-TURN INJECTION (operator, 2026-09-27): a skill is decided on EVERY
request and injected only when that turn brings new evidence for it; after
its first body, a need that comes back gets a RECALL line (its first DO
item and its first DO NOT item), never the body again. REMOVED 2026-09-27
(docs/CONSTANTS-AUDIT.md, D): the fade (FADE_TOKENS), the recall cooldown
(RECALL_EVERY_STEPS), the per-step body cap (STEP_MAX_BODIES), the per-turn
token meter (TURN_TOKENS_*), the title, tag, topic, trigger and test caps
and the description aim. A recall now comes on an EVENT only (asked,
error, phase), never the identical line twice in a row.

`AIM` values steer (the prompts are told them). `HARD` values are
enforced: the validator FAILS a skill over its hard caps rather than
rewriting it (operator, 2026-09-26: flag, do not rewrite silently).

PROHIBITIONS (operator correction, 2026-09-26): "This model responds to do
not / never so it is not strictly banned, just should be used more
sparingly." AGENTS.md "Prompting this model": at most two, each naming a
specific observed failure. `MAX_PROHIBITIONS` counts DO NOT items AND any
item whose text prohibits (never / do not / don't / must not); a skill over
it FAILS validation with the count in its reason.

Tokens are ESTIMATED at 3 characters per token -- tiers.estimate_prompt_tokens'
deliberately high rate, so a cap errs toward injecting less.
"""
from __future__ import annotations

import math
import os
import re


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


CHARS_PER_TOKEN = 3.0

# Per skill: the injected BODY (the title line and the items; the
# frontmatter is never injected).
SKILL_TOKENS_AIM = (100, 300)
SKILL_TOKENS_HARD = _i("YAMADORI_SKILL_TOKENS_HARD", 450)
MAX_ITEMS = _i("YAMADORI_SKILL_MAX_ITEMS", 6)
MIN_ITEMS = 1
# One item: the recipe corpus runs p50 166 / p90 233 characters (2,575 rows,
# 2026-09-26), so 300 keeps ~95% of it whole; longer is two items.
MAX_ITEM_CHARS = _i("YAMADORI_SKILL_MAX_ITEM_CHARS", 300)
# At most this many prohibition items per skill (see PROHIBITIONS above).
MAX_PROHIBITIONS = _i("YAMADORI_SKILL_MAX_PROHIBITIONS", 2)
# A source quote shorter than this proves nothing ("use a" is in every
# document); the builder enforces it and the prompts state it, from here
# (worker.MIN_EVIDENCE_CHARS' value; the prompt said 30 while the code
# enforced 24 until 2026-09-27).
MIN_QUOTE_CHARS = 24
MAX_DO_NOT = MAX_PROHIBITIONS             # the old name, one release

# The SKILL.md frontmatter (the Agent Skills / Hermes contract).
NAME_CHARS = 64
DESCRIPTION_CHARS = 1024                  # hard (Agent Skills, Hermes)

# Per decision (operator, 2026-09-27): NO per-turn token budget and no
# small count cap -- the skills are budgeted at input (above). The ceiling
# is a sanity bound against a pathological request (a CHOICE).
MAX_SKILLS_PER_TURN = _i("YAMADORI_SKILL_CEILING", 8)
# Items in one recall line: its first DO (or WHEN) item and its first DO
# NOT item -- the operator's own form, "remember we don't do x, we do y"
# (2026-09-27): one of each, a structure, not a count.
RECALL_ITEMS = 2
# The start-of-conversation craft index (skill_select.index_text): at most
# this many entries, each cut to INDEX_LINE_CHARS (CHOICES).
INDEX_MAX = _i("YAMADORI_SKILL_INDEX_MAX", 12)
INDEX_LINE_CHARS = 140

# Triggers: every sentence of the description (bounded by its 1,024
# characters) and every "when to use" line are kept; what the fallback
# teaches is bounded below.
MAX_LEARNED_TRIGGERS = _i("YAMADORI_SKILL_MAX_LEARNED_TRIGGERS", 40)

# Activation tests stored with every skill (skill_tests.py).
MIN_SHOULD = 2                            # request shapes that select it
MIN_SHOULD_NOT = 2                        # near misses that must not

# What counts as a prohibition inside an item's text.
PROHIBITION = re.compile(
    r"\b(?:never|do not|don't|dont|must not|mustn't|must never)\b", re.I)


def tokens(text: str) -> int:
    """Estimated tokens: high on purpose."""
    return int(math.ceil(len(text or "") / CHARS_PER_TOKEN))


def is_prohibition(item: dict | str) -> bool:
    """A DO NOT item, or any item whose text prohibits."""
    if isinstance(item, dict):
        if str(item.get("form") or "").upper() == "DO NOT":
            return True
        text = f"{item.get('situation') or ''} {item.get('text') or ''}"
    else:
        text = str(item or "")
        if re.match(r"^\s*-?\s*(?:DO NOT|DON'T)\b", text, re.I):
            return True
    return bool(PROHIBITION.search(text))


# ASSURED VOICE (operator, 2026-09-28, after pagoda-h6). The definition, in
# the operator's words: "All of our skills increase confidence and improve
# correctness; if it can't, then the line doesn't need to exist. That is
# what a skill is about: facts, proven patterns, helpful knowledge you can
# act on immediately." And on plans: "plans don't sow doubt"; "This model
# overthinks, don't give it reason to!"
#
# So what we inject -- a skill item, a plan line, a hand-off line -- states
# the current way to do a thing and the next concrete action. A line of
# these shapes is REJECTED wherever it is written, and the rejection is
# recorded (skill_builder.validate drops the item with the reason;
# skills._row_of leaves it out of what an armed skill serves;
# shomen.plan_handoff and shomen.handoff drop the line):
#   verify_first     homework: verify / check / confirm / look up an API,
#                    a prop, a README or a version, or anything "before
#                    writing / using / relying on" it
#   instability      an API, a version or an alpha that may change, differ
#                    or break; "is unstable", "is incomplete"
#   version_history  what changed between versions: deprecated, removed
#                    in, renamed, "is gone", migrate from/to, upgrading,
#                    or a version compared with another
#   caveat           be careful / be mindful / beware / watch out / keep in
#                    mind
# Running the result -- the build, the tests, the page -- is an action,
# not doubt, and passes. A version named as the version a line is FOR
# passes. A concrete pitfall that names the wrong move and the right one
# ("DO NOT: cast with as any"; "X, not Y") is GOOD content and passes
# (operator, 2026-09-28).
_VERIFY_VERB = (r"(?:verify|check|confirm|double[- ]check|re-?check"
                r"|look\s+up|consult)")
_DOC_OBJECT = (r"(?:readme|docs?|documentation|changelog|release\s+notes"
               r"|apis?|props?|signatures?|typings|\.d\.ts|versions?"
               r"|node_modules|types?\s+definitions?)")
# An imperative's start: the line's start, after a sentence or a label's
# colon, or after "first" / "always" -- never mid-clause ("define and
# check component props at compile time" is the compiler's work, not
# homework).
_START = r"(?:^|(?<=[.;:!(])\s*|\b(?:first|always)\s+)"
_VERSION_TOKEN = re.compile(
    r"(?<![\w.])(?:v(?!8\b)\d+(?:\.\d+)*(?:-[a-z]+(?:\.\d+)?)?"
    r"|\d+\.\d+\.\d+(?:-(?:alpha|beta|rc|canary)(?:\.\d+)?)?"
    r"|(?:alpha|beta|rc)\.?\d+|r1\d\d"
    r"|(?:typescript|react|three(?:\.js)?|r3f|drei|node(?:\.js)?"
    r"|c\+\+|rust|zig|python)\s\d+(?:\.\d+)?)(?![\w])", re.I)
DOUBT = (
    ("verify_first", re.compile(
        _START + _VERIFY_VERB + r"\b[^.;\n]{0,90}?\bbefore\s+(?:writ|us"
        r"|rely|call|import|depend|start|cod|implement|adopt|ship)", re.I)),
    ("verify_first", re.compile(
        _START + r"(?:re-?)?" + _VERIFY_VERB + r"\b[^.;\n]{0,60}?\b"
        + _DOC_OBJECT + r"\b", re.I)),
    ("instability", re.compile(
        r"\b(?:api|apis|exports?|signatures?|alphas?|betas?|releases?"
        r"|versions?|\d+\.\d+(?:\.\d+)?)\s+(?:\w+\s+){0,2}(?:may|might"
        r"|could)\s+(?:also\s+)?(?:still\s+)?(?:shift|change|differ|break"
        r"|not\s+work|look\s+different|behave\s+differently|be\s+(?:removed"
        r"|renamed|different|missing|incomplete|unstable)|have\s+(?:been\s+)?"
        r"(?:removed|renamed|changed))", re.I)),
    ("instability", re.compile(
        r"\b(?:is|are|remains?)\s+(?:still\s+)?(?:unstable|incomplete"
        r"|in\s+flux|not\s+(?:yet\s+)?stable)\b|\bdoes\s+not\s+mean\b[^.]"
        r"{0,40}\bworks\b|\bwill\s+change\s+between\s+versions\b"
        r"|\b(?:may|might|could)\s+(?:differ|change)\s+(?:from|between|in)"
        r"\s+(?:v?\d|versions?|releases?|alphas?)|\bsubject\s+to\s+change"
        r"|\bprovisional\b|\b(?:treat|consider)\b[^.;]{0,40}\bexperimental"
        r"\b", re.I)),
    ("version_history", re.compile(
        r"\b(?:is|are)\s+gone\b|(?<![\[\w@])deprecat\w*(?!\]\])"
        r"|\bearlier\s+(?:alphas?|versions?|releases?)\b|\bolder\s+"
        r"(?:compilers?|versions?|releases?)\b|\b(?:removed|renamed|dropped"
        r"|deleted)\s+(?:in|since|from)\b|\bmigrat\w*\s+(?:from|to|guide"
        r"|path)\b|\bupgrad\w*\s+(?:from|past|to\s+(?:v?\d|react|typescript"
        r"|three|r3f|drei|node|the\s+(?:new|latest)\s+version))"
        r"|\bupgrade\s+(?:path|guide)\b", re.I)),
    # A release that DOES something to the API: "v10 replaces priority
    # with phases", "alpha.3 deletes Canvas props", "r181 renames ...".
    ("version_history", re.compile(
        r"(?<![\w.])(?:v(?!8\b)\d+(?:\.\d+)*|(?:alpha|beta|rc)\.?\d+"
        r"|r1\d\d|\d+\.\d+(?:\.\d+)?)\s+(?:replaces|removes|deletes"
        r"|renames|deprecates|drops|splits|narrows|collapses|stops|moves"
        r"|broke|changed|made|relaxed)\b", re.I)),
    ("caveat", re.compile(
        r"\b(?:be\s+(?:careful|mindful|wary)|beware|watch\s+out"
        r"|keep\s+in\s+mind|contested)\b", re.I)),
)
# Change words that are ordinary prose unless a version is named with them.
_HISTORY_NEAR_VERSION = re.compile(
    r"\b(?:no\s+longer|previously|used\s+to|removes|deletes|renames"
    r"|renamed|drops|dropped|changed|relaxed|became)\b",
    re.I)
# Two different releases (v9 and v10, alpha.3 and alpha.5, r171 and r181)
# compared in one line: history.
_RELEASE = re.compile(r"(?<![\w.])(?:v(?!8\b)\d+|(?:alpha|beta|rc)\.?\d+|r1\d\d)"
                      r"(?![\w])", re.I)


def doubt(text: str) -> str | None:
    """The ASSURED VOICE rule's reason for rejecting `text`, or None. A
    concrete pitfall -- the wrong move and the right one ("X, not Y";
    "DO NOT: cast with as any") -- is good content and passes."""
    t = str(text or "")
    for why, pat in DOUBT:
        if pat.search(t):
            return why
    if _VERSION_TOKEN.search(t) and _HISTORY_NEAR_VERSION.search(t):
        return "version_history"
    if len({r.lower().replace(".", "") for r in _RELEASE.findall(t)}) >= 2:
        return "version_history"
    return None


def summary() -> dict:
    """For the dashboard: every limit, and the label that goes with them."""
    return {"label": "choices, not measurements (skill_limits.py says why)",
            "chars_per_token": CHARS_PER_TOKEN,
            "skill_tokens_aim": list(SKILL_TOKENS_AIM),
            "skill_tokens_hard": SKILL_TOKENS_HARD,
            "max_items": MAX_ITEMS, "max_item_chars": MAX_ITEM_CHARS,
            "max_prohibitions": MAX_PROHIBITIONS,
            "name_chars": NAME_CHARS,
            "description_chars": DESCRIPTION_CHARS,
            "max_skills_per_turn": MAX_SKILLS_PER_TURN,
            "index_max": INDEX_MAX,
            "min_should": MIN_SHOULD, "min_should_not": MIN_SHOULD_NOT,
            "max_learned_triggers": MAX_LEARNED_TRIGGERS}
