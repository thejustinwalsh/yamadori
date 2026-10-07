#!/usr/bin/env python
"""`reasoning_effort` as the product dial: how hard to try, in one standard field.

WHY OVERLOAD AN EXISTING FIELD

Everything this stack adds -- skills, the MCP host's lookups, a concept
seed -- costs latency and tokens, and none of it is worth paying for on
"rename this variable". So a caller needs a way to say how hard to try.

The obvious move is a new parameter. It is also the wrong one: a new parameter
is configuration, no existing client sends it, and the entire premise here is
that a client adds one OpenAI-compatible model and gets the stack without
configuring anything.

`reasoning_effort` is already in the chat-completions schema, clients already
send it, and it already means "how hard should this try". Overloading it to
select the augmentation set as well as the thinking budget costs no new
surface and reads correctly to someone who has never heard of this proxy.

THE HONEST PROBLEM WITH PRESETS

A tier bundles two things that vary independently: thinking budget and which
augmentations run. If `high` beats `low`, that does not say whether the gain
came from the extra reasoning or from what we added, and a bundled win is
not attributable.

That is fine for a PRODUCT, where a person wants one dial, and wrong for a
MEASUREMENT. So the resolver takes explicit overrides: a benchmark can hold
the budget fixed and vary only the augmentations, or the reverse, and recover
the factorial design the presets deliberately collapse.

WHAT EACH TIER IS: COGNITIVE EFFORT, NOT SPEND

This is the axis, and getting it wrong is easy. On self-hosted hardware there
is no bill: the tokens are free, the GPU is already paid for, and a tier that
spends three times as much of it costs nothing anyone is counting. Ordering
these by token spend -- which this docstring used to do -- describes an
accountant's concern that does not exist here.

What the dial actually moves is HOW HARD THE SYSTEM THINKS:

  minimal   the model alone, thinking OFF (the fast tier)
  low       the model alone, with its own reasoning (the baseline)
  medium    + the MCP host's package lookups (skills: off at every tier
            since 2026-09-29)
  high      + a concept seed on the conversation's first user turn
  xhigh     the same as high
  max       the same as xhigh, at the template's xhigh thinking effort

The second brain -- the fix-up repair, fan-out and deep thinking, which
`high` and up used to add -- and the library definitions `medium` added were
REMOVED 2026-09-29 (docs/REMOVED.md; the way back is commit e360d37).

Nothing here claims a tier is BETTER. That is what the benchmarks are for, and
until they have run on a stack whose parts all work, these are effort tiers
with a plausible ordering.

  minimal   thinking off, the vendor's instruct sampling, nothing of ours.
  low       the model AS IT SHIPS: thinking at medium, none of our
            augmentation. The benchmark baseline (see the note on TIERS).
  medium    adds the MCP host's package lookups (and skills, where a header
            turns them on).
  high      adds the concept seed.
  xhigh     high's bundle.
  max       xhigh's bundle at the template's xhigh effort.
"""
from __future__ import annotations

import os

# Accepted spellings, including the ones clients actually send. OpenAI ships
# "minimal" | "low" | "medium" | "high"; Anthropic-flavoured clients sometimes
# send "xhigh"; some send "none" or "off" meaning do not think.
ALIASES = {
    # "off" and "none" pick `minimal`, which since 2026-09-23 (operator) DOES
    # switch the model's thinking off -- the fast tier, with the vendor's
    # instruct sampling. See the note on TIERS for what that costs.
    "none": "minimal", "off": "minimal", "minimal": "minimal",
    "low": "low", "lo": "low",
    "medium": "medium", "med": "medium", "default": "medium", "": "medium",
    "high": "high", "hi": "high",
    "xhigh": "xhigh", "x-high": "xhigh", "very_high": "xhigh",
    "max": "max", "maximum": "max", "ultra": "max",
}

ORDER = ["minimal", "low", "medium", "high", "xhigh", "max"]

# CORRECTED 2026-09-22. This block used to say a per-request reasoning budget
# is ignored and only the server flag counts. That was measured with the wrong
# field name (`reasoning_budget`). The running build honours
# `reasoning_budget_tokens` (alias `thinking_budget_tokens`,
# server-common.cpp:1354 at 9a9394a) -- with 20000 set, reasoning ran to
# 19,987 tokens. See docs/CONSTRAINTS.md 1a.
#
# `thinks` says whether thinking is on; `effort` is a STYLE sentence in the
# template, not a length control (xhigh thought less than low on 2 of 3
# measured prompts). Token budgets are one global rule, `budget()` below --
# there are no per-tier floors any more.
# THERE IS NO TOOL BUDGET DIAL, AND THERE SHOULD NOT BE ONE.
#
# A `hops` key used to live here, one number per tier, described as "BOTH told
# to the model and enforced in the loop". It was neither useful nor, in the
# second half, true:
#
#   NEVER ENFORCED. Both tool loops in proxy.py read MAX_TOOL_HOPS, never the
#   tier. A `low` request was told "at most 4 tool-calling turns" while its
#   real ceiling was 12. We were stating a number to the model that nothing
#   checked.
#
#   TREATING A SYMPTOM. The benchmark arm that "spent twelve generations on a
#   problem the other arms answered in one" was not overspending, it was
#   RETRYING A BROKEN TOOL. Tools returned empty strings on failure, so the
#   model could not tell "nothing matched" from "this is broken" and called
#   again. Tool results now state the situation, whether it is retryable as a
#   fact, and a remedy with an owner. That is what ends a loop.
#
# A tool loop now ends when the model stops asking for tools, when the
# conversation no longer fits its KV share (proxy.context_full), or at the
# vendor's tool-turn cap, MAX_TOOL_TURNS = 10 (PrismML agenticMaxTurns;
# operator, 2026-09-23) in proxy.py. The cap LANDS -- tools withdrawn,
# answer requested -- and is recorded (x_yamadori.tool_turns), so hitting it
# with repeated empty calls reads as a tool defect.
#
# THE FLAGS BELOW MEAN "ALLOWED", NOT "ON" (2026-09-22, docs/SELECTION-BUILD.md
# step 5). `skills` is the most a caller at this tier can get; the skills
# system decides per request what fires, and never exceeds it. Only a flag
# set in X-Yamadori-Features forces a system on or off, for experiments.
# `seed` (2026-09-29): a concept seed is drawn and injected on the
# conversation's first user turn (proxy.prepare) -- the tiers where every
# second-brain job carried one until those jobs were removed.
#
# REMOVED 2026-09-29 (docs/REMOVED.md): `retrieval` (the library-definitions
# injection and LIBRARY USE), `check_code` and `repair` (the code check's
# notes and the fix-up repair), `fanout`, `investigate` (deep thinking) and
# `delegate`. A header that still sends one is ignored (from_header).

TIERS = {
    # SKILLS ARE OFF AT EVERY TIER (operator, 2026-09-29: "Stop skills until
    # we have a good skill injector." -- "Skills are still valuable we just
    # haven't found the unlock yet. TBD."). `skills` is False below, so
    # nothing of the skills system reaches the model's context: no skill
    # bodies, no recall lines, no yama_recall_craft offer or craft index.
    # The skills CODE (pipeline and serving) stays; a header forcing
    # {"skills": true} still reaches it (how the offline suites test it).
    #
    # `minimal` IS THINKING OFF; `low` IS THE BASELINE (operator, 2026-09-23).
    #
    # `minimal` is the fast tier: thinking off, the vendor's instruct sampling
    # (VENDOR_SAMPLING_INSTRUCT: temp 0.7, top_p 0.80, top_k 20, presence 1.5
    # -- PrismML's card, and the thinking-off line in snailium/bonsai2-8gb and
    # sudoingX's MTP card), none of our augmentation. Two recorded results
    # argue against expecting much from it, and both stay visible here:
    #   - SPEED, n=3 problems: thinking OFF spent 4,363 completion tokens over
    #     270s, thinking ON 1,770 over 110s, both 3/3 -- off was SLOWER.
    #   - INJECTION, n=5 per arm, one naive payload (config.yaml): thinking
    #     OFF exfiltrated 5/5, ON (temp 1.0) 0/5. The least injection-resistant
    #     tier; never the one for an agent reading untrusted input.
    #
    # `low` is the model AS IT SHIPS -- its own thinking at medium, none of
    # our augmentation (skills, the MCP tools, the seed). It is the
    # honest benchmark baseline (it used to be `minimal`, with thinking on at
    # low effort; switching thinking OFF would cripple the thing we compare
    # against, and any arm beating it would bank thinking-beats-no-thinking).
    "minimal": {
        "thinks": False, "effort": "low",
        "skills": False, "seed": False,
        "mcp_tools": False,
        "why": "thinking off, the vendor's instruct sampling, nothing of ours",
    },
    "low": {
        "thinks": True, "effort": "medium",
        "skills": False, "seed": False,
        "mcp_tools": False,
        "why": "the model as it ships -- its own thinking, none of "
               "our augmentation. The benchmark baseline.",
    },
    "medium": {
        "thinks": True, "effort": "medium",
        "skills": False, "seed": False,
        "mcp_tools": True,
        "why": "the MCP host's package lookups (skills: off at every tier "
               "since 2026-09-29)",
    },
    # Medium effort, not xhigh (operator, 2026-09-23): xhigh thinking is
    # confined to `max`. Same evidence as the `xhigh` tier note below.
    "high": {
        "thinks": True, "effort": "medium",
        "skills": False, "seed": True,
        "mcp_tools": True,
        "why": "the MCP host's package lookups and a concept seed on the "
               "conversation's first user turn",
    },
    # `xhigh` and `max` differ ONLY in the effort sent (operator, 2026-09-23).
    # Every augmentation is allowed on both; `xhigh` sends the model `medium`
    # effort, `max` sends `xhigh`. (Since 2026-09-29 `xhigh` is `high`'s
    # bundle: deep thinking, what set it apart, was removed.) So the pair is
    # the effort-matched comparison: whatever separates them is the template's xhigh effort and
    # nothing else. Why it is worth measuring: sudoingX's bonsai2-small-gpu
    # (kernel/reasoning_effort.md, greedy, 3 tasks x 2 runs) found template
    # xhigh ran away inside <think> at small caps where medium finished, and
    # PrismML's card recommends medium "for shorter responses and a balance of
    # speed and accuracy". No result in this repo says xhigh helps.
    "xhigh": {
        "thinks": True, "effort": "medium",
        "skills": False, "seed": True,
        "mcp_tools": True,
        "why": "everything, at medium thinking: the effort-matched pair to "
               "max",
    },
    "max": {
        "thinks": True, "effort": "xhigh",
        "skills": False, "seed": True,
        "mcp_tools": True,
        "why": "everything at xhigh thinking",
    },
}

# THE FEATURE MATRIX (operator, 2026-09-24): what actually RUNS at each tier,
# derived from TIERS and nowhere else. The AGENTS.md and README tier tables,
# the dashboard's tier ladder (/dash/api/tiers `features`) and
# mcp/test_tier_docs.py all read it, so a copy that drifts fails a test.
FEATURE_COLUMNS = ("thinking", "skills", "MCP tools", "images",
                   "concept seed")


def features(name: str, doc: str = "AGENTS.md") -> dict:
    """{column: cell} for one tier. `doc` README.md says whether the tier
    thinks (no effort mapping on show, operator 2026-09-23); AGENTS.md
    states the effort actually sent."""
    t = TIERS[name]
    if doc == "README.md":
        thinking = "on" if t.get("thinks") else "off"
    else:
        thinking = safe_effort(t.get("effort")) if t.get("thinks") else "off"
    return {
        "thinking": thinking,
        "skills": "yes" if t.get("skills") else "–",
        # The MCP host's package lookups (switch `mcp_tools`, mcp/mcp_host.py).
        "MCP tools": "yes" if t.get("mcp_tools") else "–",
        "images": "yes" if images_offered(t) else "–",
        # Drawn once, on the conversation's first user turn (2026-09-29).
        "concept seed": "yes" if t.get("seed") else "–",
    }


def feature_row(name: str, doc: str = "AGENTS.md") -> list[str]:
    f = features(name, doc)
    return [f[c] for c in FEATURE_COLUMNS]


# TOOL-TURN LIMIT PER CONTEXT (operator, 2026-09-23). PrismML's Bonsai-demo
# bounds a tool loop at agenticMaxTurns = 10; `max` gets 20, because at max
# the stack is asked to work hardest. Neither number is measured --
# 10 is the vendor demo's default, 20 is the operator's; every response
# records the limit it ran under (x_yamadori.tool_turns).
TOOL_TURNS = int(os.environ.get("YAMADORI_MAX_TOOL_TURNS", "10"))
TOOL_TURNS_MAX = int(os.environ.get("YAMADORI_MAX_TOOL_TURNS_MAX", "20"))


def tool_turn_limit(tier) -> int:
    """Tool turns one context may take at this tier (a tier dict or name)."""
    name = tier if isinstance(tier, str) else (tier or {}).get("name")
    return TOOL_TURNS_MAX if name == "max" else TOOL_TURNS


# A deployment may cap what callers can ask for, because `max` is unbounded
# enough that an open endpoint should not offer it by default.
CEILING = os.environ.get("YAMADORI_TIER_CEILING", "max")
DEFAULT = os.environ.get("YAMADORI_TIER_DEFAULT", "medium")



# WHAT THE TEMPLATE WILL ACTUALLY ACCEPT -- READ FROM THE TEMPLATE, NOT RECALLED.
#
# Three layers disagree about this and only the innermost one matters:
#   OpenAI's schema    minimal | low | medium | high
#   llama.cpp's CLI    also advertises xhigh and max
#   THE chat template  raises a Jinja exception -> HTTP 500 on anything else
#
# THIS WAS A HARDCODED SET, AND IT WENT STALE SILENTLY. It said {minimal, low,
# medium, high, xhigh, ""}, "probed directly", on an earlier build. The
# template the fixed PrismML fork serves accepts exactly ('xhigh', 'medium',
# 'low') and raises on everything else, "" included:
#
#     Unexpected reasoning effort high. Supported types are xhigh (default),
#     medium, and low.
#
# So every `high`-tier request -- fan-out, the tier sold as "it considers
# alternatives" -- was an instant 500, 0.0s, re-tried once and reported as a
# 502. Found 2026-09-22 by the live stack suite, not by any offline test,
# because every offline test asserted against this same stale set.
#
# Now the set is parsed out of the served template's own guard clause
# (`not in ('xhigh', 'medium', 'low')`) via llama-swap's /upstream/<model>/props,
# cached per process. The fallback, used only if the server cannot be asked,
# is what that template says today.
EFFORT_ORDER = ["minimal", "low", "medium", "high", "xhigh"]
FALLBACK_EFFORTS = ("low", "medium", "xhigh")
_accepted: tuple[str, ...] | None = None
# per model (MAX MODE, mcp/max_mode.py): each main model's served template is read once, and only while it serves
_accepted_of: dict[str, tuple[str, ...]] = {}


def efforts_in_template(tpl: str) -> tuple[str, ...]:
    """The efforts a chat template's guard clause accepts (`reasoning_effort
    not in ('xhigh', 'medium', 'low')`), or () when it has none. One parser
    for the served template and its pinned fixture (mcp/served_fixture.py)."""
    import re
    m = re.search(r"reasoning_effort\s+not\s+in\s+\(([^)]*)\)", tpl or "")
    return tuple(re.findall(r"'([a-z]+)'", m.group(1))) if m else ()


def accepted_efforts(refresh: bool = False) -> tuple[str, ...]:
    """The reasoning_effort values the served chat template accepts."""
    global _accepted
    import max_mode
    model = max_mode.current(os.environ.get("YAMADORI_MODEL", "bonsai"))
    if max_mode.ENABLED:
        if model in _accepted_of and not refresh:
            return _accepted_of[model]
        if max_mode.blocks(model) and max_mode.bound_model() != model:
            # MAX MODE: reading /props would load a model that is off the card; the template's set is read once it
            # serves (both templates accept the same three today: docs/FLASH-NEXT.md section 2)
            return FALLBACK_EFFORTS
    elif _accepted is not None and not refresh:
        return _accepted
    found: tuple[str, ...] = ()
    upstream = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
    # A READ NEVER LOADS THE MODEL. /upstream/<model>/props makes llama-swap
    # start a model that is off the card, and a dashboard view (/dash/api/
    # tiers -> safe_effort) did exactly that on 2026-09-28 11:47, in the
    # Flash-Next gate's window: the gate's guard saw bonsai start and killed
    # its arm. llama-swap's GET /running never loads; when it says the model
    # is not loaded, the fallback is answered and NOT kept, so the template
    # is read once the model serves.
    try:
        import gpu_room
        rows = gpu_room.running(upstream)
    except Exception:                                            # noqa: BLE001
        rows = None
    # /running unreadable is NOT a licence to ask /upstream (2026-09-30: the
    # dashboard's /dash/api/tiers must never load a model): llama-swap busy or
    # restarting cannot say what is loaded, and asking /upstream/<model>/props
    # then would start it. The fallback is answered and not kept.
    if rows is None or model not in {
            str(r.get("model")) for r in rows
            if str(r.get("state", "ready")) != "stopped"}:
        return FALLBACK_EFFORTS
    try:
        import json
        import urllib.request
        with urllib.request.urlopen(f"{upstream}/upstream/{model}/props",
                                    timeout=5) as r:
            tpl = json.load(r).get("chat_template") or ""
        found = efforts_in_template(tpl)
    except Exception:                                            # noqa: BLE001
        found = ()
    _accepted = found or FALLBACK_EFFORTS
    if max_mode.ENABLED and found:
        _accepted_of[model] = _accepted
    return _accepted


def safe_effort(value: str) -> str:
    """An effort string the template will not throw on.

    The intent is preserved by rounding UP to the nearest accepted level on
    EFFORT_ORDER (a caller asking for `high` gets more thinking, not less),
    and down to the highest accepted level only above the top of the scale.
    Aliases (`max`, `none`, ...) are resolved first.
    """
    v = (value or "").strip().lower()
    v = {"max": "xhigh", "maximum": "xhigh", "ultra": "xhigh",
         "x-high": "xhigh", "none": "minimal", "off": "minimal",
         "": "xhigh"}.get(v, v)
    ok = accepted_efforts()
    if v in ok:
        return v
    if v not in EFFORT_ORDER:
        v = "medium"
    rank = EFFORT_ORDER.index(v)
    for cand in EFFORT_ORDER[rank:]:
        if cand in ok:
            return cand
    return max(ok, key=lambda e: EFFORT_ORDER.index(e)
               if e in EFFORT_ORDER else -1)


# Kept for callers and tests that read it: the live accepted set.
ACCEPTED_EFFORTS = set(FALLBACK_EFFORTS)


def normalise(value) -> str:
    if value is None:
        return DEFAULT
    return ALIASES.get(str(value).strip().lower(), DEFAULT)


def images_offered(tier: dict) -> bool:
    """Is yama_generate_image offered? Always, wherever an image server is
    configured (proxy.image_tools checks that): it is a capability, not a
    gate or an augmentation of the answer (operator, 2026-09-23). Every tier,
    `minimal` included, and header-forced benchmark arms too."""
    return True


def resolve(body: dict, overrides: dict | None = None) -> dict:
    """The tier for this request, after aliasing, capping and overrides.

    `overrides` is how a benchmark breaks the bundle apart: pass
    {"seed": false} to a high-tier request and the extra reasoning is
    measured without the seed. Without that, a preset comparison can only
    ever report that the bundle helped, never which part of it did.
    """
    effort = body.get("reasoning_effort")
    if effort is None and isinstance(body.get("reasoning"), dict):
        # OpenRouter's object form, {"reasoning": {"effort": "high"}}: same
        # dial, the other spelling many clients send.
        effort = body["reasoning"].get("effort")
    name = normalise(effort)
    cap = normalise(CEILING)
    if ORDER.index(name) > ORDER.index(cap):
        name = cap
    tier = dict(TIERS[name])
    tier["name"] = name
    if overrides:
        tier.update(overrides)
        tier["overridden"] = sorted(overrides)
    return tier


# THE TOKEN BUDGET: DERIVED FROM THE CONTEXT SPLIT. docs/CONSTRAINTS.md s.3,
# mcp/budget.py.
#
# Every token a request makes -- prompt, THINKING and answer -- occupies cells
# of the one KV pool the server reserved at load (-c 147456, four slots, one
# unified cache). The pool is split by role, as decided by the operator:
#
#     main     5/8   the conversation
#     helper   3/8   the helper role (then: ONE second brain at a time;
#                    removed 2026-09-29 -- today the child slot's side work)
#
# So a request's thinking room is not a free-floating number. It is what its
# share leaves after its prompt and its answer allowance:
#
#     window    = share(role) / n
#     answer    = max(client max_tokens, A_MIN)
#     thinking  = window - prompt - answer      (never below MIN_THINKING)
#     upstream.max_tokens              = thinking + answer
#     upstream.reasoning_budget_tokens = thinking
#
# Within its share the model is let cook: nothing measured says a shorter
# thought is better. The share is what keeps concurrent requests from
# overflowing the pool, which is the only thing a limit is for here. The
# caller's max_tokens stays an ANSWER allowance (the empty-reply bug was
# thinking eating a TOTAL number: summarize_text at 900 returned nothing).
#
# BUDGET_MESSAGE makes a forced close end in an answer instead of mid-thought
# (one measured run without it: 999 reasoning tokens, 5,000 tokens of thinking
# in the answer channel, finish=length, no answer; with it, 2 of 2 clean).
A_MIN = int(os.environ.get("YAMADORI_ANSWER_MIN", "2048"))
MIN_THINKING = 1024
# THINKING CAPS (operator, 2026-09-25: "this model overthinks pretty quickly").
# Octopus v0b-V0 (88 agent steps, n=1 run): reasoning was 96% of the output;
# median step 730 completion tokens, p90 3,713, max 24,620 (48 min to choose a
# file search). A 4,096 cap would have cut 6 steps and ~38K tokens (~70 min at
# ~9 tok/s). CHOICES, not measurements of quality under the cap:
#   AGENT_STEP_THINKING  main's thinking on a request routed `agent_step`
#                        (a step in the client's own tool loop). A user turn
#                        and compaction keep their budgets.
#   HELPER_THINKING      a helper-role request (the skills pipeline's model
#                        stages, summaries). The second brain's per-job caps
#                        (JOB_THINKING's fixup / investigate / plan /
#                        alternative / tiebreak rows) went with its jobs,
#                        2026-09-29.
# The helper SHARE is 0.30 now (mcp/budget.py). A benchmark's
# X-Yamadori-Features {"reasoning_cap": N} still wins over all of these.
AGENT_STEP_THINKING = int(os.environ.get("YAMADORI_AGENT_STEP_THINKING", "6144"))
# The cap BY WHAT THE STEP ANSWERS (#53: AGENT_STEP_THINKING_READ 2,048 after
# a read / search / inspect, AGENT_STEP_THINKING_ERROR 6,144 after an error,
# progress.step_kind, switch step_thinking) was REMOVED 2026-09-27
# (docs/CONSTANTS-AUDIT.md: chosen from one Octopus run). Every agent step
# thinks at most AGENT_STEP_THINKING.
# USER_TURN_THINKING: every other main turn (a question, a task start, a code
# request). ~3x the longest NATURAL finish measured (682-2,826 tokens, 5 of 7
# runs, docs/CONSTRAINTS.md 1b); the other 2 of 3 on one prompt ran past
# 20,000 without stopping, and a 95-fold verbatim loop is on record
# (docs/TRANSCRIPT-REVIEW-2026-09-23.md). Was 12,288 (operator, 2026-09-25).
# 20,480 since 2026-09-27 (operator: "Yes bump user turn, and I trust their
# numbers more than ours"): professorpalmer/bonsai-ada-surgery's measured
# recipe for this model at medium effort, a 20,480-token thinking budget with a
# force-close message (docs/QUALITY.md there: Killy's HumanEval-164 grid,
# medium 161/164; docs/ENGINES.md "Token floor and thinking budget"). Their grid
# is a total-cap measurement, not our force-closed cap: the source, not a
# measurement of this proxy. The 0.6 nudge now fires at 12,288.
USER_TURN_THINKING = int(os.environ.get("YAMADORI_USER_TURN_THINKING", "20480"))
# All five are 1.5x their first values (operator, 2026-09-25: "1.5 those
# numbers above and dial it in"): 4,096 / 8,192 / 4,096 and jobs 2,048 /
# 4,096 / 8,192. V0's watcher counts cap hits to dial them in.
HELPER_THINKING = int(os.environ.get("YAMADORI_HELPER_THINKING", "6144"))
# A helper job's own cap, by job name (model.shape passes it as the step
# cap). EMPTY since the second brain's jobs were removed (2026-09-29): a
# skill pipeline job (skill.<purpose>) has no row and is capped at
# HELPER_THINKING.
JOB_THINKING: dict[str, int] = {}
# THE THINKING NUDGE (operator, 2026-09-25). Once thinking has used NUDGE_AT
# of its budget, the served fork (llamacpp-sudoingx-bonsai2, common/
# reasoning-budget.cpp) forces NUDGE_MESSAGE into the reasoning at the next
# token that carries a newline, then lets the model keep thinking; the nudge's
# own tokens count against the budget. It fires at most once per generation,
# never when thinking ends first, and not at all when fewer than ~64 tokens
# plus the nudge would remain -- the hard stop (BUDGET_MESSAGE) then works as
# before. The nudge OPENS a short converging summary in the model's own voice
# that the model completes, and the hard-stop message points back at that
# summary, so a forced close lands on a finished conclusion rather than
# mid-thought. Both texts are UNMEASURED WORDING -- prompts, to be A/B'd --
# kept positive (AGENTS.md: prohibitions degrade this model). NUDGE_MESSAGE
# ends on a newline so the model's summary starts cleanly on the next line.
# NUDGE_AT 0.8 is the operator's choice, not a measurement. Off with
# YAMADORI_THINKING_NUDGE=0. A llama-server without the patch ignores both
# fields (unknown request keys are not evaluated), so the proxy can ship
# first; BUDGET_MESSAGE then points at a summary that was never opened, so
# switch the binary with it.
# Reworded 2026-09-25 (operator): "What I know, and the single best next
# step:" asked for a plan, and at 60% the model re-planned and switched
# approach instead of finishing (live, n=1). This matches the original
# hard-stop line's voice: a heads-up to wrap up, then the stop.
# Reworded again the same day (operator): in the model's own voice -- Qwen
# thinking opens "The user ..." and acts with "Let me ..." (8 samples of
# Bonsai's reasoning: "Let me" 57, "So" 50, "I'll" 24) -- and the one reason
# to keep going is honesty: round 3 (no nudge) hit the hard stop and then
# claimed a verification it never did.
NUDGE_MESSAGE = ("\n\nThe user is waiting for a response. Let me go with the "
                 "best answer I've already worked out, and only keep thinking "
                 "if that answer would be misleading.\n")
NUDGE_AT = 0.6   # operator 2026-09-25 (was 0.8); live round 5, n=1: converged, no hard stop
# THE AGENT-STEP NUDGE (#53; APPROVED by the operator as written, 2026-09-26). The
# nudge above stays the operator's for every other turn. A step in the
# client's own tool loop ends in a CALL, not an answer, and after a read the
# model deliberated to the nudge and then asked to read again (T4); the
# located-defect runs stated the fix and never applied it (#50, T1). This
# variant is the same voice -- "The user is waiting", "Let me", the honesty
# clause kept -- and names the action: the call it has worked out, the edit
# when it knows the fix. UNMEASURED WORDING. Off: YAMADORI_AGENT_STEP_NUDGE=0
# or X-Yamadori-Features {"step_nudge": false}: agent steps then get
# NUDGE_MESSAGE as before.
AGENT_STEP_NUDGE_MESSAGE = (
    "\n\nThe user is waiting for a response. Let me make the call I've "
    "already worked out -- the edit itself, if I know the fix -- and only "
    "keep thinking if that call would be misleading.\n")
def helper_nudge(job: str | None) -> str | None:
    """The nudge text for a helper job of its own, or None (NUDGE_MESSAGE).
    None for every job since the research jobs' nudge
    (HELPER_NUDGE_MESSAGE, investigate / plan) was removed with them,
    2026-09-29; the skills pipeline still asks (skill_pipeline)."""
    return None


# The hard stop keeps its ORIGINAL line (operator, 2026-09-25: the variant
# "Thinking limit reached. I'll act on the summary above now." was tried
# live, n=1, and the model then acted on an invented path; "the original
# text ... seemed to work better").
BUDGET_MESSAGE = ("\n\nThinking budget reached. I will stop deliberating and "
                  "write the final answer now.\n")


def nudge_on() -> bool:
    """YAMADORI_THINKING_NUDGE, read per call (default on)."""
    v = os.environ.get("YAMADORI_THINKING_NUDGE")
    return (v if v is not None else "1").strip().lower() not in (
        "0", "", "off", "false", "no")


def nudge_fields(message: str | None = None, at: float | None = None) -> dict:
    """The nudge's request fields, or {} when it is off. Only for a request
    that thinks with a finite budget (the fork ignores it otherwise).
    `message`: the text to force (AGENT_STEP_NUDGE_MESSAGE for an agent
    step); NUDGE_MESSAGE when None."""
    if not nudge_on():
        return {}
    # YAMADORI_NUDGE_AT overrides the fraction (read per call; 0 < x < 1),
    # for the operator's 0.6 vs 0.8 vs off comparison (2026-09-25).
    # `at`: a model's token profile (mcp/tier_models.py) names its own fraction; the env var still wins (the
    # operator's comparison knob).
    default_at = at if at is not None and 0 < float(at) < 1 else NUDGE_AT
    try:
        at = float(os.environ.get("YAMADORI_NUDGE_AT", "") or default_at)
    except ValueError:
        at = default_at
    if not 0 < at < 1:
        at = default_at
    return {"reasoning_budget_nudge": message or NUDGE_MESSAGE,
            "reasoning_budget_nudge_at": at}


def window_model_of(model: str | None) -> str | None:
    """The model whose OWN window budgets this request, when it is not the conversation's: a table model that is not
    a main one -- the helper (bonsai-a4000) an internal job was routed to (max_mode.internal_model; coordinator
    2026-09-30: "a job routed to bonsai-a4000 is budgeted against its fitted window, not the main line"). None
    otherwise: the main models' windows follow the request's model as before (budget.model_window)."""
    if not model:
        return None
    try:
        import max_mode
        import tier_models
        t = tier_models.table()
        if t.profiles_on and model in t.models and not max_mode.is_main(model) and t.window(model):
            return model
    except Exception:                                            # noqa: BLE001
        return None
    return None


def _shares(model: str | None = None) -> dict:
    """{main, helper, pool, capped}. `pool` is the most ONE request may
    occupy: the whole pool under the split, the main cap under the cap layout
    (budget.py THE CAP LAYOUT: nothing runs past the VRAM line), where
    `capped` is True and a running child does not shrink it (the child swaps
    in while its conversation's main pauses). `model` (window_model_of): a
    helper server's own window instead of the main line."""
    try:
        import budget as _budget
        b = _budget.budgets(model=model) if model else _budget.budgets()
        return {"main": b["main"], "helper": b["helper"],
                "pool": b.get("window") or b["pool"],
                "capped": b.get("layout") == "cap"}
    except Exception:                                            # noqa: BLE001
        return {"main": 114688, "helper": 49152, "pool": 163840,
                "capped": False}


def estimate_prompt_tokens(body: dict) -> int:
    """A deliberately HIGH estimate of the prompt's tokens, from its size.

    ~3 characters per token over messages and tools. Over-estimating only
    lowers the thinking room a little; under-estimating could overflow the
    share. The exact count would cost a /tokenize round trip per request.
    """
    import json as _json
    import image_input
    n = 0
    for key in ("messages", "tools"):
        v = body.get(key)
        if key == "messages" and v:
            # an image part passed to the main model (vision.normalise) counts
            # as the projector's tokens for it (image_input.image_tokens),
            # never as its base64 text
            v, image_tok = image_input.without_image_bytes(v)
            n += 3 * image_tok
        if v:
            try:
                n += len(_json.dumps(v, ensure_ascii=False))
            except (TypeError, ValueError):
                n += len(str(v))
    return n // 3


def budget(client_max_tokens, thinks: bool = True, cap: int | None = None,
           role: str = "main", share_n: int = 1,
           prompt_tokens: int = 0, step_cap: int | None = None,
           nudge: str | None = None, profile: dict | None = None,
           window_model: str | None = None) -> dict:
    """The upstream token fields for one request, derived from its share.
    `window_model` (window_model_of): the helper server whose own window this
    request is budgeted against.

    `cap` overrides the derived thinking room for ONE request -- the
    benchmark sets it through X-Yamadori-Features {"reasoning_cap": N} to
    measure whether longer thinking helps. Clamped to [MIN_THINKING, pool].

    `profile` (resolved by apply() from the model's token profile,
    mcp/tier_models.py): {answer, turn_min, force_close, nudge_at, sized,
    job_caps}. `step_cap` and `nudge` then already carry the profile's
    values for the route; `sized` (a client request) means the client's
    max_tokens does not size the turn -- the profile's answer allowance does.
    Without a profile (no table) everything is as before.
    """
    prof = profile or {}
    try:
        answer = int(client_max_tokens or 0)
    except (TypeError, ValueError):
        answer = 0
    floor = int(prof.get("answer") or A_MIN)
    answer = floor if prof.get("sized") else max(answer, floor)
    turn_min = int(prof.get("turn_min") or 0)
    if not thinks:
        if prof.get("sized"):
            s = (_shares(window_model) if window_model else _shares())
            window = s["helper" if role == "helper" else "main"] // max(int(share_n), 1)
            # THE PROFILE SIZES A CLIENT'S TURN (operator, 2026-09-29: "This is stuff the user and the harness will
            # just get wrong"): thinking off, the turn is the whole room the window leaves, never below the
            # profile's turn floor -- as a thinking turn's max_tokens is its whole room.
            return {"max_tokens": max(window - int(prompt_tokens or 0), answer, turn_min)}
        return {"max_tokens": max(answer, turn_min)}
    s = (_shares(window_model) if window_model else _shares())
    window = s["helper" if role == "helper" else "main"] // max(int(share_n), 1)
    thinking = max(window - int(prompt_tokens or 0) - answer, MIN_THINKING)
    # THE STANDING CAPS LIMIT THINKING ONLY (operator, 2026-09-25). The
    # helper's hops always, main's agent steps when the proxy passes step_cap;
    # never below MIN_THINKING. max_tokens stays the whole room the share
    # leaves, so the ANSWER -- a whole-file write_file is ~7K tokens (Octopus
    # game.js, 709 lines) -- is never squeezed by a thinking cap: a capped
    # step sending thinking + A_MIN = 6,144 in total would cut such a write
    # off even with no thinking at all.
    room = thinking
    # MAX MODE (mcp/max_mode.py): the max model thinks as its card says, not under Bonsai's caps. Qwen3.8-Flash-Next's
    # card (Qwen/Qwen3.8-Flash-Next @ de4b8e4d, README "Best Practices": up to 262,144 tokens of reasoning, 131,072 of
    # answer) -- the whole room the share leaves is the budget. USER_TURN_THINKING, AGENT_STEP_THINKING,
    # JOB_THINKING and HELPER_THINKING were derived on Bonsai (Octopus v0b, bonsai-ada-surgery) and do not apply.
    # The nudge still fires at its fraction of whatever budget is sent. A benchmark's reasoning_cap still wins.
    # With a PROFILE (the table, mcp/tier_models.py) the caps are already the model's own: apply() resolved
    # step_cap from it, and `job_caps` says whether Bonsai's second-brain caps apply to this model.
    import max_mode
    if profile is not None:
        if role == "helper" and prof.get("job_caps") and HELPER_THINKING > 0 and not step_cap:
            thinking = min(thinking, max(HELPER_THINKING, MIN_THINKING))
    elif max_mode.at_max():
        step_cap = None
    elif role == "helper" and HELPER_THINKING > 0 and not step_cap:
        thinking = min(thinking, max(HELPER_THINKING, MIN_THINKING))
    if step_cap:
        thinking = min(thinking, max(int(step_cap), MIN_THINKING))
    total = max(room + answer, turn_min)
    if cap is not None:
        # A benchmark's header cap keeps its measured meaning: thinking +
        # answer is the whole allowance.
        try:
            thinking = min(max(int(cap), MIN_THINKING), s["pool"])
            total = thinking + answer
        except (TypeError, ValueError):
            pass
    return {"max_tokens": total,
            "reasoning_budget_tokens": thinking,
            "reasoning_budget_message": prof.get("force_close") or BUDGET_MESSAGE,
            **nudge_fields(nudge, prof.get("nudge_at"))}


# THE COMPACTION BUDGET (operator, 2026-09-24). A client's compaction
# (selection.utility_kind == "compaction") is served on the conversation it
# summarises (mcp/compaction.py) and THINKS at that conversation's own effort
# -- at every effort, medium included -- with the conversation's own
# sampling; thinking is off only where the conversation itself runs with it
# off (compaction.prefix_fields; interim defaults of 2026-09-24 until
# docs/COMPACTION-RESEARCH.md's eval runs). Its thinking budget is what its
# own window leaves -- window - prompt - answer, never below MIN_THINKING --
# the one budget rule (budget() above) applied to the compaction's window
# (compaction_budget's `thinking_tokens`). COMPACTION_THINKING (2,048, "5 of the 7
# self-finished thinking runs" on coding prompts, NOT a compaction
# measurement) was REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md). Its answer
# allowance is at least
# COMPACTION_BUDGET, "4-5K" by the operator's call, 5,120 unless set. NOT
# MEASURED as sufficient: the corpus's 12 Hermes compaction answers ran
# 7,733-41,702 characters (median ~20,700), one of them (35,425) repetitive
# -- at 3.5-4 characters a token, 7 of the 12 were over 5,120 tokens. Watch
# finish_reason on compactions; the env var raises it.
#
# The client's stated target is honoured, bounded only by the pool: never
# below the budget (a summary cut short lands a budget notice in the
# client's own history, and Hermes then discards it and compacts again).
# Hermes sends NO max_tokens (context_compressor.py: "NEVER add a max_tokens
# wire cap on the summary call") but states "Target ~N tokens" in the prompt,
# N = 20% of the compacted turns within [2,000, min(5% of the window,
# 10,000)] -- up to ~12,192 at this pool; proxy._serve_compaction reads that
# as the client's value. (Until 2026-09-24 a fixed 2x ceiling cut it.)
#
# WHERE THE ROOM COMES FROM. The pool is unified: a compaction's new cells --
# its summary, plus whatever prefix its slot did not already hold -- can
# draw on the helper's 3/8 whenever no second brain is running. So its
# window is the POOL less an active helper's share, read once
# (admission.helper_active) and never waited for. Only when prompt + answer
# do not fit that window does it fall back to what is left of the main share,
# and the record says so. The prompt is tiers.estimate_prompt_tokens' HIGH
# estimate (chars / 3), so a fallback is reported early rather than late.
COMPACTION_BUDGET = int(os.environ.get("YAMADORI_COMPACTION_BUDGET", "5120"))


def compaction_budget(client_max_tokens, prompt_tokens: int,
                      helper_active: int = 0,
                      shares: dict | None = None) -> dict:
    """The answer allowance for one compaction, and the record of why:
    {budget, cap, client_max_tokens, answer, prompt_estimate, main_share,
    pool, helper_active, window, room, fits, draws_on_spare,
    thinking_tokens}. `thinking_tokens`: what the window it runs in leaves after the prompt and the
    answer, never below MIN_THINKING (the one budget rule)."""
    s = shares or _shares()
    b = max(int(COMPACTION_BUDGET), A_MIN)
    # No fixed ceiling (2026-09-24): the client's target, bounded by the pool
    # below.
    cap = s["pool"]
    try:
        client = int(client_max_tokens or 0)
    except (TypeError, ValueError):
        client = 0
    answer = min(max(client, b), cap) if client > 0 else b
    prompt = max(int(prompt_tokens or 0), 0)
    active = max(int(helper_active or 0), 0)
    window = s["pool"] - (s["helper"] if active and not s.get("capped")
                          else 0)
    rec = {"budget": b, "cap": cap, "client_max_tokens": client or None,
           "prompt_estimate": prompt, "main_share": s["main"],
           "pool": s["pool"], "helper_active": active, "window": window}
    if prompt + answer <= window:
        rec.update(answer=answer, fits=True,
                   room=("the main cap: nothing runs past the VRAM line"
                         if s.get("capped") else
                         "pool: no second brain running, its share is spare"
                         if not active else
                         "pool less the running second brain's share"),
                   draws_on_spare=prompt + answer > s["main"])
        rec["thinking_tokens"] = max(window - prompt - answer, MIN_THINKING)
    else:
        rec.update(answer=max(min(answer, s["main"] - prompt), A_MIN),
                   fits=False, draws_on_spare=False,
                   room=(f"main share: prompt ~{prompt} + answer {answer} "
                         f"exceed the {window}-token window"
                         + (" with a second brain running" if active else "")))
        rec["thinking_tokens"] = max(s["main"] - prompt - rec["answer"],
                              MIN_THINKING)
    return rec


# VENDOR SAMPLING, ENFORCED (operator, 2026-09-23: "any vendor settings, the
# proxy needs to enforce"). The served model is PrismML's Ternary Bonsai 2
# 27B, a Qwen3.8-27B derivative, and its card gives Qwen's thinking-mode
# values (https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf; the GGUF
# header carries the same temp 1.0 / top_p 0.95 / top_k 20). Before this,
# nothing in mcp/ set sampling: a client's temperature passed straight
# through (the domain harness sent 0.0, greedy -- off-spec for a thinking
# model), and fan-out hard-coded 0.2-0.7 per variant. apply() is the one
# door for client AND internal requests (model.py), so it is written here
# and overrides whatever the caller sent; x_yamadori.sampling records what
# was asked for and what was applied.
VENDOR_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                   "min_p": 0.0, "presence_penalty": 0.0,
                   "repeat_penalty": 1.0}
# The same card's instruct (non-thinking) values. No tier turns thinking off
# today; this is what such a request would get.
VENDOR_SAMPLING_INSTRUCT = {"temperature": 0.7, "top_p": 0.80, "top_k": 20,
                            "min_p": 0.0, "presence_penalty": 1.5,
                            "repeat_penalty": 1.0}


def enforce_sampling(body: dict, thinks: bool = True,
                     values: dict | None = None) -> tuple[dict, dict]:
    """(sampling fields to send, the record for x_yamadori.sampling).

    The record is {enforced: {...}, client_overridden: {key: client value}}
    -- only keys the caller set to something else appear in the second.
    `values`: a model's token profile sampling (mcp/tier_models.py)."""
    want = dict(values or (VENDOR_SAMPLING if thinks else VENDOR_SAMPLING_INSTRUCT))
    over = {k: body[k] for k in want
            if k in body and body[k] is not None and body[k] != want[k]}
    return want, {"enforced": dict(want), "client_overridden": over}


# THE TOKEN PROFILE (operator, 2026-09-29: "Token settings, thinking efforts, etc. ... it is mostly about tuning
# how big of a turn we allow, and what the thinking budget should be. This is stuff the user and the harness will
# just get wrong, the proxy makes this atomic agent style"). With the tier -> model table on (mcp/tier_models.py),
# every request gets its MODEL's one profile whatever the client sends: the effort sent to the template, the
# thinking budget and nudge by route (a user turn / a step of the client's own tool loop), the answer allowance,
# the turn floor, the force-close message and the sampling. The client's reasoning_effort picks the tier and so the
# model; its max_tokens does not size the turn. A benchmark's X-Yamadori-Features (`effort`, `reasoning_cap`) still
# wins -- the header is how a paired measurement varies one thing. Without the table nothing here changes.
ROUTES = ("user_turn", "agent_step", "job", "none")


def _route_of(role: str, step_cap: int | None, turn: str | None) -> str:
    """Which of the profile's routes a request is. `turn` when the caller names it; else read from what it passed:
    proxy.prepare passes AGENT_STEP_THINKING for an agent step, USER_TURN_THINKING for any other turn, None for a
    side call or a compaction; a helper-role caller (model.shape) passes its job's cap with role helper."""
    if turn in ROUTES:
        return turn
    if role == "helper":
        return "job"
    if step_cap is None:
        return "none"
    if step_cap == AGENT_STEP_THINKING:
        return "agent_step"
    if step_cap == USER_TURN_THINKING:
        return "user_turn"
    return "job"


def resolve_profile(body: dict, tier: dict, role: str = "main", step_cap: int | None = None,
                    nudge: str | None = None, turn: str | None = None) -> dict | None:
    """The model's token profile for this request, resolved to the values apply() and budget() send, and the record
    for x_yamadori.sampling.profile. None when the table is off (one model: every constant as before)."""
    import max_mode
    import tier_models
    if not tier_models.profiles_on():
        return None
    model = max_mode.current(body.get("model") if max_mode.is_main(body.get("model")) else None)
    p = tier_models.profile(model)

    def val(k):
        return (p.get(k) or {}).get("value")

    def cls(k):
        return {"class": (p.get(k) or {}).get("class"), "source": (p.get(k) or {}).get("source")}
    route = _route_of(role, step_cap, turn)
    forced = set(tier.get("overridden") or [])
    effort = tier.get("effort")
    effort_from = "tier"
    if val("effort") and "effort" not in forced:
        effort, effort_from = val("effort"), "profile"
    elif "effort" in forced:
        effort_from = "header"
    cap = step_cap
    msg = nudge
    cap_key = "job_caps"
    nudge_key = "nudge_user_turn"
    if route == "user_turn":
        cap, cap_key = val("user_turn_thinking"), "user_turn_thinking"
        msg = val("nudge_user_turn") or nudge
    elif route == "agent_step":
        cap, cap_key = val("agent_step_thinking"), "agent_step_thinking"
        # the agent-step nudge only where the caller chose it (switch step_nudge); else the user-turn text
        if nudge:
            nudge_key = "nudge_agent_step"
        msg = val(nudge_key) or nudge
    elif route == "job" and not val("job_caps"):
        cap = None
    client = bool(body.get("_client")) and role == "main"
    rec = {"model": model, "table": tier_models.table().source, "route": route,
           "effort": {"sent": effort, "from": effort_from, **cls("effort")},
           "thinking_cap": {"value": cap, **cls(cap_key)},
           "nudge": {"at": val("nudge_at"), "chars": len(msg or "") or None, **cls(nudge_key)},
           "answer_tokens": {"value": val("answer_tokens"), **cls("answer_tokens")},
           "turn_min_tokens": {"value": val("turn_min_tokens"), **cls("turn_min_tokens")},
           "force_close": {"chars": len(val("force_close") or ""), **cls("force_close")},
           "sampling": cls("sampling_thinking" if tier.get("thinks") else "sampling_instruct"),
           "client_sized": client}
    return {"effort": effort, "step_cap": cap, "nudge": msg,
            "sampling": val("sampling_thinking") if tier.get("thinks") else val("sampling_instruct"),
            "budget": {"answer": val("answer_tokens"), "turn_min": val("turn_min_tokens"),
                       "force_close": val("force_close"), "nudge_at": val("nudge_at"), "sized": client,
                       "job_caps": bool(val("job_caps"))},
            "record": rec}


def _client_effort(body: dict):
    if body.get("reasoning_effort") is not None:
        return body.get("reasoning_effort")
    r = body.get("reasoning")
    return r.get("effort") if isinstance(r, dict) else None


def apply(body: dict, tier: dict, role: str = "main",
          share_n: int = 1, step_cap: int | None = None,
          nudge: str | None = None, turn: str | None = None) -> dict:
    """Write the tier's sampling settings into the upstream request.

    The augmentation flags are NOT written here -- they are read by the proxy,
    which decides what to run. Only what the model server itself understands
    is set: the effort, whether it thinks, and the token budget, derived from
    the request's share of the pool (role, share_n) and its own prompt.

    With the tier -> model table on, the model's TOKEN PROFILE (above,
    resolve_profile) supplies the effort, the caps, the nudge, the answer
    allowance, the turn floor, the force-close and the sampling; `turn`
    names the route when the caller knows it ("user_turn" / "agent_step").
    """
    out = dict(body)
    # `max_completion_tokens` is OpenAI's newer name for the same answer
    # allowance. Read it when `max_tokens` is absent and never pass it on:
    # left in the body it would reach the model server beside our own
    # max_tokens and bypass the budget rule.
    client_max = body.get("max_tokens")
    if client_max is None:
        client_max = body.get("max_completion_tokens")
    out.pop("max_completion_tokens", None)
    prof = resolve_profile(body, tier, role=role, step_cap=step_cap, nudge=nudge, turn=turn)
    if prof is not None:
        step_cap, nudge = prof["step_cap"], prof["nudge"]
    fields, record = enforce_sampling(body, bool(tier["thinks"]),
                                      prof["sampling"] if prof else None)
    out.update(fields)
    if prof is not None:
        # x_yamadori.sampling.profile: the profile applied (values, their class and source) and what the client
        # asked for
        record["profile"] = dict(prof["record"], client={"max_tokens": client_max,
                                                         "reasoning_effort": _client_effort(body)})
        out["_profile"] = prof["budget"]
    out["_sampling"] = record
    out.update(budget(client_max, tier["thinks"],
                      tier.get("reasoning_cap"), role=role, share_n=share_n,
                      prompt_tokens=estimate_prompt_tokens(body),
                      step_cap=step_cap, nudge=nudge,
                      profile=prof["budget"] if prof else None,
                      window_model=window_model_of(body.get("model"))))
    if tier["thinks"]:
        # The answer allowance, kept: max_tokens is now the whole room (the
        # caps limit thinking only), so rebudget cannot recover it by
        # subtraction any more.
        floor = int(((prof or {}).get("budget") or {}).get("answer") or A_MIN)
        if prof is not None and prof["budget"].get("sized"):
            out["_answer"] = floor
        else:
            try:
                out["_answer"] = max(int(client_max or 0), floor)
            except (TypeError, ValueError):
                out["_answer"] = floor
    if step_cap:
        # rebudget() re-derives the budget for later hops; it keeps the cap.
        out["_step_cap"] = int(step_cap)
    if nudge:
        # rebudget() keeps the agent-step nudge for later hops.
        out["_nudge"] = nudge
    if tier.get("reasoning_cap") is not None and tier["thinks"]:
        # The same for a benchmark's X-Yamadori-Features reasoning_cap: live
        # 2026-09-25 it was lost on the main loop's rebudget (128,244 sent
        # for a cap of 1,200).
        out["_reasoning_cap"] = int(tier["reasoning_cap"])
    # The chat template reads enable_thinking from chat_template_kwargs; the
    # top-level field is kept because the proxy's own loops read it
    # (rebudget). Before 2026-09-23 only the top-level field was written, and
    # config.yaml's llama-swap filter forced chat_template_kwargs
    # enable_thinking=true on every request, so no request could turn
    # thinking off.
    ctk = dict(out.get("chat_template_kwargs") or {})
    ctk["enable_thinking"] = bool(tier["thinks"])
    out["chat_template_kwargs"] = ctk
    if tier["thinks"]:
        out["reasoning_effort"] = safe_effort(prof["effort"] if prof else tier["effort"])
        out["enable_thinking"] = True
    else:
        out["enable_thinking"] = False
        out.pop("reasoning_effort", None)
        out.pop("reasoning_budget_tokens", None)
        out.pop("reasoning_budget_message", None)
        out.pop("reasoning_budget_nudge", None)
        out.pop("reasoning_budget_nudge_at", None)
    return out


def rebudget(payload: dict, role: str = "main", share_n: int = 1) -> dict:
    """Re-derive an ALREADY-SHAPED payload's budget for its prompt as it
    is now: each later hop of a turn (its prompt includes the hops before
    it). share_n > 1 splits a share n ways, for n concurrent samples;
    nothing in the proxy runs those. A payload shaped under a token profile
    keeps it (`_profile`)."""
    if not payload.get("enable_thinking", True) or \
            "reasoning_budget_tokens" not in payload or \
            payload.get("_fixed_budget"):
        # _fixed_budget: a compaction keeps the smallest thinking budget it
        # was given (proxy._serve_compaction), not the share's.
        return dict(payload)
    out = dict(payload)
    answer = out.get("_answer") or max(
        int(out.get("max_tokens") or 0)
        - int(out.get("reasoning_budget_tokens") or 0), A_MIN)
    prof = out.get("_profile")
    if prof is not None:
        # the answer is already the profile's; a later hop is not re-sized from a client value
        prof = dict(prof, sized=False, answer=answer)
    out.update(budget(answer, True, out.get("_reasoning_cap"), role=role,
                      share_n=share_n,
                      prompt_tokens=estimate_prompt_tokens(out),
                      step_cap=out.get("_step_cap"),
                      nudge=out.get("_nudge"), profile=prof))
    return out


def describe() -> str:
    lines = ["reasoning_effort selects how hard this tries:"]
    for n in ORDER:
        t = TIERS[n]
        bits = []
        if t["skills"]:
            bits.append("skills")
        if t.get("mcp_tools"):
            bits.append("MCP tools")
        if t.get("seed"):
            bits.append("concept seed")
        lines.append(f"  {n:<8} effort {t['effort']:<7} "
                     f"{', '.join(bits) or 'nothing'} -- {t['why']}")
    return "\n".join(lines)

if __name__ == "__main__":
    if os.environ.get("YAMADORI_SERVED_FIXTURE") == "1":
        # mcp/test_tiers.py runs this self-test OFFLINE: the pinned /props
        # (mcp/served_fixture.py), never the live server's.
        import sys
        import served_fixture
        served_fixture.pin(sys.modules[__name__])
    print(describe())
    print()
    for v in ("xhigh", "HIGH", "none", None, "nonsense", "med"):
        t = resolve({"reasoning_effort": v})
        print(f"  {str(v):<10} -> {t['name']:<8} "
              f"effort={t['effort']:<7} max_tokens={apply({}, t)['max_tokens']:<6} "
              f"seed={t['seed']}")
    t = resolve({"reasoning_effort": "high"}, overrides={"seed": False})
    # `budget` was renamed to `floor` and this line was not updated, so the
    # self-test raised KeyError instead of printing -- which is why nobody
    # noticed. A self-test that crashes is worse than none: it looks like
    # coverage and provides none.
    print(f"\n  high with the seed forced off (factorial arm): "
          f"seed={t['seed']}, "
          f"overridden={t.get('overridden')}")


# A true A/B needs the augmentations toggled at a FIXED reasoning level.
# The presets deliberately move both at once, which is right for a product
# dial and useless for attribution: if `max` beats `minimal`, the bundle won
# and nobody can say which half did it.
#
# These are the two arms that isolate it. Same effort, same floor, same
# template, same preamble -- only the injected skills differ.
AB_ARMS = {
    "aug_off": {"skills": False},
    "aug_on": {"skills": True},
}


def ab(effort: str = "medium", arm: str = "aug_off") -> dict:
    """One arm of the augmentation A/B, at a reasoning level you choose."""
    body = {"reasoning_effort": effort}
    return resolve(body, overrides=AB_ARMS[arm])


def from_header(value: str | None) -> dict | None:
    """Feature overrides from `X-Yamadori-Features`, for experiments.

    A header rather than a body field: the body is the OpenAI schema and a
    benchmark's knobs do not belong in it, where they would also become part
    of the cached prompt. Returns None when absent, so normal traffic is
    untouched and the tier presets decide as usual.
    """
    if not value:
        return None
    import json as _json
    try:
        d = _json.loads(value)
    except (ValueError, TypeError):
        return None
    # The header is client-controlled. `[1]` or `"x"` used to reach d.items()
    # and raise, and `{"floor": "big"}` passed through to apply(), where
    # max(int, str) raised -- either way one malformed header crashed the
    # request. A value of the wrong type is dropped like an unknown key.
    if not isinstance(d, dict):
        return None
    # Every flag set here is FORCED on or off; flags left out are ALLOWED by
    # the tier and decided per request by mcp/selection.py --
    # `tier["overridden"]` is how it tells them apart. The flags of removed
    # features (retrieval, investigate, fanout, delegate, check_code, repair;
    # 2026-09-29) are unknown keys now: dropped like any other.
    # `utility` forces the client-utility-call decision (proxy.utility_of,
    # selection.utility_call) either way: true -> the bare model at minimal,
    # false -> a task turn whatever the request looks like.
    # `hints` is the flag's old name (skills replaced hints, 2026-09-26):
    # benchmarks still send it, so it is read as `skills` for ONE release.
    if "hints" in d and "skills" not in d:
        d = dict(d, skills=d["hints"])
    d.pop("hints", None)
    types_ok = {"skills": bool, "seed": bool, "effort": str,
                "reasoning_cap": int, "utility": bool,
                # IDLE CLEAR's threshold for one request, in seconds: the
                # live `slots` test's override. The proxy honours it for a
                # TEST account only, and then clears only that account's
                # own idle conversations (mcp/slots.py clear_idle).
                "idle_clear_s": int,
                # ONE CONVERSATION's hold for one request, in seconds: the
                # live suite's override (it opens a new conversation every
                # few seconds). Honoured for a TEST account only, and only
                # against that account's own owner (mcp/slots.py _hold_for).
                "primary_hold_s": int}
    # The overthinking switches (BEHAVIOURS below), each forced either way.
    types_ok.update({k: bool for k in BEHAVIOURS})
    # The package skills channel's rendering (mcp/package_skills.py MODES).
    types_ok["package_skills_mode"] = str
    out = {k: v for k, v in d.items()
           if k in types_ok and isinstance(v, types_ok[k])
           and not (types_ok[k] is int and isinstance(v, bool))}
    return out or None


# THE SWITCHES (operator, 2026-09-26; docs/research/OVERTHINKING.md,
# docs/SELF-IMPROVEMENT-LOG.md #50-#56). Every behaviour listed is ON by
# default and can be switched off -- for a paired control arm -- by
# X-Yamadori-Features {"<name>": false} or its environment variable (=0). A
# header forces it either way and wins over the variable; `allow` names the
# tier flag that must be true when no header forces it (None: wherever the
# mechanism it changes runs).
# REMOVED 2026-09-27 (operator: task-targeted steering in prompts; skills
# are the channel): progress_note (#50), unchanged_read (#54) and
# plan_entry_first (#55), with the text they injected.
# REMOVED 2026-09-29 with the features they switched (docs/REMOVED.md):
# seed_frame, helped_needs_change, plan_prompt, deep_tool_hop,
# continue_stated_step, auto_triggers, verify_directive.
#   work_log_reinject   the work log after a compaction of a conversation
#                       whose session id did not change (#41 made the old
#                       new-key link unreachable for Hermes; #54)
#   step_nudge         the agent-step nudge wording (#53)
#   slot_release       not an overthinking switch, the same mechanism: the
#                       transient slot is emptied after each side call
#                       (mcp/slots.py RELEASE; #58, an idle slot's cache
#                       slows every decode)
#   idle_clear          the same, for an idle CONVERSATION's pinned slot:
#                       cleared for an active request after IDLE_CLEAR_S,
#                       pin and ledger kept (mcp/slots.py IDLE CLEAR; #59)
BEHAVIOURS = {
    "work_log_reinject": ("YAMADORI_WORK_LOG_REINJECT", None),
    "step_nudge": ("YAMADORI_AGENT_STEP_NUDGE", None),
    "slot_release": ("YAMADORI_SLOT_RELEASE", None),
    "idle_clear": ("YAMADORI_IDLE_CLEAR", None),
    #   restore_reasoning  PAST REASONING IS RESTORED (operator, 2026-09-27:
    #                   "Keeping thinking across turns seems useful, fuck
    #                   Hermes, Hermes can do whatever it wants."). The
    #                   ledger records the slot's own reasoning for every
    #                   assistant turn it delivers and puts it back into each
    #                   past turn whose reasoning the client dropped; a
    #                   client's echo is kept as sent. The served template
    #                   renders every past turn's think block
    #                   (`preserve_thinking` undefined = on), so with it
    #                   restored each request EXTENDS the slot again
    #                   (proxy.ledger_restore; AGENTS.md "Past reasoning is
    #                   restored"). Off (YAMADORI_RESTORE_REASONING=0 or the
    #                   header): the 2026-09-24 pass-through.
    "restore_reasoning": ("YAMADORI_RESTORE_REASONING", None),
    #   mcp_tools       THE MCP TOOLS (operator, 2026-09-28: "the proxy
    #                   provides MCP servers to the model zero-config for
    #                   every harness, starting with PackageLens"): the
    #                   yama_* tools the proxy's MCP host backs
    #                   (mcp/mcp_host.py) and their one system line, offered
    #                   on a conversation's first request and kept. Allowed
    #                   where the tier's `mcp_tools` flag is set -- `medium`
    #                   and up (the operator's "offered at medium and up by
    #                   default") -- so `medium` with {"skills": false} is the
    #                   model plus our MCP tools only. YAMADORI_MCP_TOOLS=0 or
    #                   the header turns it off; {"mcp_tools": true} forces it
    #                   on at any tier.
    "mcp_tools": ("YAMADORI_MCP_TOOLS", "mcp_tools"),
    #   package_skills  THE PACKAGE SKILLS CHANNEL (operator, 2026-10-06:
    #                   "If the model uses some of our other tools or mcps
    #                   this may be a good point to skill up"): the proven
    #                   skills of a package ride in the result of the MCP
    #                   package tool that named it, and evidence-based
    #                   triggers between turns (mcp/package_skills.py).
    #                   OFF BY DEFAULT until its probe passes (THE TOOL
    #                   RECIPE rule 7; bench/mcp/lookup_probe.py arms
    #                   package-*): YAMADORI_PACKAGE_SKILLS=1 or the header
    #                   {"package_skills": true} turns it on; allowed where
    #                   the tier's `mcp_tools` flag is set (medium and up).
    #                   The rendering is `package_skills_mode` (inject |
    #                   router | both; header or YAMADORI_PACKAGE_SKILLS_MODE).
    "package_skills": ("YAMADORI_PACKAGE_SKILLS", "mcp_tools"),
    #   preread         THE EXPERT FILE'S PRE-READ (mcp/preread.py; AGENTS.md
    #                   "Flash-Next's first prompt"): a model whose tier-table
    #                   row says `preread` has its llama-server's mmapped
    #                   expert file read into the OS file cache when it is
    #                   swapped in and when its working set was trimmed.
    #                   YAMADORI_PREREAD=0 or the header turns it off.
    "preread": ("YAMADORI_PREREAD", None),
    #   preread_overlap on: the read starts as the swap request goes out and
    #                   overlaps the load; off: the read, then the load (the
    #                   live check measures both; overlap is the default).
    "preread_overlap": ("YAMADORI_PREREAD_OVERLAP", None),
}
# The switches that are OFF unless their variable (=1) or a header turns
# them on; every other switch is on unless switched off. None since
# deep_tool_hop went with deep thinking (2026-09-29).
OFF_BY_DEFAULT: frozenset = frozenset({"package_skills"})


def _env_on(var: str, default: bool = True) -> bool:
    v = os.environ.get(var)
    return (v if v is not None else ("1" if default else "0")
            ).strip().lower() not in ("0", "", "off", "false", "no")


def behaviour(tier: dict | None, name: str) -> bool:
    """Is overthinking switch `name` on for this request (BEHAVIOURS)?"""
    return behaviour_source(tier, name)[0]


def behaviour_source(tier: dict | None, name: str) -> tuple[bool, str]:
    """(on, why): `header` when X-Yamadori-Features forced it, `env` when
    its variable switched it off, `tier` when the tier does not allow it,
    `default` otherwise."""
    env, allow = BEHAVIOURS[name]
    tier = tier or {}
    if name in (tier.get("overridden") or []) and name in tier:
        return bool(tier[name]), "header"
    if not _env_on(env, default=name not in OFF_BY_DEFAULT):
        return False, ("default" if name in OFF_BY_DEFAULT
                       and os.environ.get(env) is None else "env")
    if allow and not tier.get(allow):
        return False, "tier"
    return True, "default"


def behaviour_of_header(features, name: str) -> tuple[bool, str]:
    """behaviour_source for a switch read BEFORE the tier is resolved (the
    ledger restore in proxy.prepare): only the header's forced values."""
    feats = from_header(features) or {}
    return behaviour_source(dict(feats, overridden=sorted(feats)), name)


def behaviours(tier: dict | None) -> dict:
    """{name: {on, source}} for x_yamadori.progress.switches."""
    return {n: dict(zip(("on", "source"), behaviour_source(tier, n)))
            for n in BEHAVIOURS}
