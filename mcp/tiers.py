#!/usr/bin/env python
"""`reasoning_effort` as the product dial: how hard to try, in one standard field.

WHY OVERLOAD AN EXISTING FIELD

Everything this stack adds -- retrieval, skills (formerly hints), fan-out, a second
investigating context -- costs latency and tokens, and none of it is worth
paying for on "rename this variable". So a caller needs a way to say how hard
to try.

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
came from the extra reasoning or from the retrieval, and a bundled win is not
attributable.

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
  medium    + it recalls skills, reads library definitions, notes broken writes
  high      + a second model repairs broken code and compares a second answer
  xhigh     + a second mind investigates before it answers (medium effort)
  max       the same as xhigh, at the template's xhigh thinking effort

Each step adds a kind of thinking, not a quantity of tokens. That is also why
`reasoning_effort` is the right field to overload: it already means "how hard
should this try", and every step above is a different answer to that question.

The one real cost is WALL CLOCK, because the caller waits. Measured on the
second context: 737s to 1,423-1,752s. Streaming the investigation back as
`reasoning_content` is what makes that wait watchable rather than blank, which
is a product decision about the experience of waiting, not about spend.

Nothing here claims a tier is BETTER. That is what the benchmarks are for, and
until they have run on a stack whose parts all work, these are effort tiers
with a plausible ordering.

  minimal   thinking off, the vendor's instruct sampling, nothing of ours.
  low       the model AS IT SHIPS: thinking at medium, none of our
            augmentation. The benchmark baseline (see the note on TIERS).
  medium    adds library definitions (a tail injection), skills (formerly
            recipe hints) and the write check (a note, no fix).
  high      adds the second brain's repair (writes and answers), fan-out
            (sequential, via the second brain) and the static addendum.
  xhigh     adds the second investigating context, at medium effort.
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
# operator, 2026-09-23) in proxy.py and shomen.py. The cap LANDS -- tools
# withdrawn, answer requested -- and is recorded (x_yamadori.tool_turns), so
# hitting it with repeated empty calls reads as a tool defect.
#
# THE FLAGS BELOW MEAN "ALLOWED", NOT "ON" (2026-09-22, docs/SELECTION-BUILD.md
# step 5). `hints`, `fanout` and `investigate` are the most a caller at this
# tier can get; mcp/selection.py decides per request whether each one fires,
# and never exceeds them. Only a flag set in X-Yamadori-Features forces a
# system on or off, for experiments. `retrieval` is gated separately, by
# domains.tool_admission.
#
# `check_code` and `repair` (2026-09-23, mcp/code_check.py; operator
# 2026-09-24). Since 2026-09-24 there is no check_code TOOL on main: the
# proxy checks the code itself. `check_code` means client writes (write_file,
# patch, ...) are checked before they are forwarded and a note says what was
# found (mcp/tool_code.py) -- `medium` and up. `repair` means the second
# brain's fixup job repairs what does not parse, in client writes and in a
# final answer's code, and the static addendum describes it -- `high`,
# `xhigh` and `max`. Neither needs an index, so neither follows `retrieval`;
# a header forces either one either way.
TIERS = {
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
    # our retrieval, hints, checks, fan-out or second context. It is the
    # honest benchmark baseline (it used to be `minimal`, with thinking on at
    # low effort; switching thinking OFF would cripple the thing we compare
    # against, and any arm beating it would bank thinking-beats-no-thinking).
    "minimal": {
        "thinks": False, "effort": "low",
        "retrieval": False, "hints": False, "fanout": 1, "investigate": False,
        "check_code": False, "repair": False,
        "why": "thinking off, the vendor's instruct sampling, nothing of ours",
    },
    "low": {
        "thinks": True, "effort": "medium",
        "retrieval": False, "hints": False, "fanout": 1, "investigate": False,
        "check_code": False, "repair": False,
        "why": "the model as it ships -- its own thinking, none of "
               "our augmentation. The benchmark baseline.",
    },
    "medium": {
        "thinks": True, "effort": "medium",
        "retrieval": True, "hints": True, "fanout": 1, "investigate": False,
        "check_code": True, "repair": False,
        "why": "skills that apply (silent when none clearly does), library "
               "definitions for a library question, and a note when a "
               "client write does not parse",
    },
    # Medium effort, not xhigh (operator, 2026-09-23): xhigh thinking is
    # confined to `max`. Same evidence as the `xhigh` tier note below.
    "high": {
        "thinks": True, "effort": "medium",
        "retrieval": True, "hints": True, "fanout": 3, "investigate": False,
        "check_code": True, "repair": True,
        "why": "the second brain: code that does not parse is repaired "
               "(writes and answers), code answers are compared with an "
               "independent second approach, and the addendum says so",
    },
    # `xhigh` and `max` differ ONLY in the effort sent (operator, 2026-09-23).
    # Every augmentation is allowed on both; `xhigh` sends the model `medium`
    # effort, `max` sends `xhigh`. So the pair is the effort-matched
    # comparison: whatever separates them is the template's xhigh effort and
    # nothing else. Why it is worth measuring: sudoingX's bonsai2-small-gpu
    # (kernel/reasoning_effort.md, greedy, 3 tasks x 2 runs) found template
    # xhigh ran away inside <think> at small caps where medium finished, and
    # PrismML's card recommends medium "for shorter responses and a balance of
    # speed and accuracy". No result in this repo says xhigh helps.
    "xhigh": {
        "thinks": True, "effort": "medium",
        "retrieval": True, "hints": True, "fanout": 3, "investigate": True,
        "check_code": True, "repair": True,
        "why": "everything, deep thinking included, at medium thinking: the "
               "effort-matched pair to max",
    },
    "max": {
        "thinks": True, "effort": "xhigh",
        "retrieval": True, "hints": True, "fanout": 3, "investigate": True,
        "check_code": True, "repair": True,
        "why": "everything at xhigh thinking; deep thinking's hand-off "
               "becomes this answer's thinking",
    },
}

# THE FEATURE MATRIX (operator, 2026-09-24): what actually RUNS at each tier,
# derived from TIERS and nowhere else. The AGENTS.md and README tier tables,
# the dashboard's tier ladder (/dash/api/tiers `features`) and
# mcp/test_tier_docs.py all read it, so a copy that drifts fails a test.
FEATURE_COLUMNS = ("thinking", "library help", "skills", "code check",
                   "fan-out", "deep thinking", "addendum", "images",
                   "concept seed")


def features(name: str, doc: str = "AGENTS.md") -> dict:
    """{column: cell} for one tier. `doc` README.md says whether the tier
    thinks (no effort mapping on show, operator 2026-09-23); AGENTS.md
    states the effort actually sent."""
    t = TIERS[name]
    fan = int(t.get("fanout") or 1)
    if doc == "README.md":
        thinking = "on" if t.get("thinks") else "off"
    else:
        thinking = safe_effort(t.get("effort")) if t.get("thinks") else "off"
    second_brain = bool(t.get("repair") or fan > 1 or t.get("investigate"))
    return {
        "thinking": thinking,
        # A tail injection of held-symbol definitions (proxy.
        # _library_definitions), not tools on main (2026-09-24).
        "library help": "definitions" if t.get("retrieval") else "–",
        "skills": "yes" if t.get("hints") else "–",
        # medium: client writes checked, a note; high and up: repaired by
        # the second brain.
        "code check": ("repair" if t.get("repair") else
                       "note" if t.get("check_code") else "–"),
        "fan-out": "1" if fan <= 1 else f"up to {fan}",
        "deep thinking": "allowed" if t.get("investigate") else "–",
        "addendum": "yes" if t.get("repair") else "–",
        "images": "yes" if images_offered(t) else "–",
        # Every second-brain job carries one (operator, 2026-09-24).
        "concept seed": "yes" if second_brain else "–",
    }


def feature_row(name: str, doc: str = "AGENTS.md") -> list[str]:
    f = features(name, doc)
    return [f[c] for c in FEATURE_COLUMNS]


# TOOL-TURN LIMIT PER CONTEXT (operator, 2026-09-23). PrismML's Bonsai-demo
# bounds a tool loop at agenticMaxTurns = 10; `max` gets 20, because at max
# the stack is asked to investigate hardest. Each context counts its own: the
# main loop has its limit, and EVERY deep-thinking run has its own on top.
# Fan-out candidates (B, C) make no tool calls. Neither number is measured --
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


def accepted_efforts(refresh: bool = False) -> tuple[str, ...]:
    """The reasoning_effort values the served chat template accepts."""
    global _accepted
    if _accepted is not None and not refresh:
        return _accepted
    found: tuple[str, ...] = ()
    try:
        import json
        import re
        import urllib.request
        upstream = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
        model = os.environ.get("YAMADORI_MODEL", "bonsai")
        with urllib.request.urlopen(f"{upstream}/upstream/{model}/props",
                                    timeout=5) as r:
            tpl = json.load(r).get("chat_template") or ""
        m = re.search(r"reasoning_effort\s+not\s+in\s+\(([^)]*)\)", tpl)
        if m:
            found = tuple(re.findall(r"'([a-z]+)'", m.group(1)))
    except Exception:                                            # noqa: BLE001
        found = ()
    _accepted = found or FALLBACK_EFFORTS
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


def check_code_offered(tier: dict) -> bool:
    """Are client writes checked (mcp/tool_code.py)? The tier's `check_code`
    (`medium` and up), or anywhere the fixup runs (`repair`); a header that
    forces `check_code` is exactly what the header says (see the note above
    TIERS). The name is kept from when this offered a tool."""
    if "check_code" in (tier.get("overridden") or []):
        return bool(tier.get("check_code"))
    return bool(tier.get("check_code")) or repair_on(tier)


def images_offered(tier: dict) -> bool:
    """Is generate_image offered? Always, wherever an image server is
    configured (proxy.image_tools checks that): it is a capability, not a
    gate or an augmentation of the answer (operator, 2026-09-23). Every tier,
    `minimal` included, and header-forced benchmark arms too. Deep thinking
    gets it as well (proxy.deep_thinking_tools), for mockups and designs."""
    return True


def repair_on(tier: dict) -> bool:
    """Does the second brain repair code that does not parse -- a client
    write, or a final answer's code (shomen.run("fixup"))?"""
    return bool(tier.get("repair"))


def resolve(body: dict, overrides: dict | None = None) -> dict:
    """The tier for this request, after aliasing, capping and overrides.

    `overrides` is how a benchmark breaks the bundle apart: pass
    {"fanout": 1} to a high-tier request and the extra reasoning is measured
    without the extra sampling. Without that, a preset comparison can only
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
#     helper   3/8   ONE second brain (deep thinking, and fan-out's extra
#                    candidates, one at a time -- mcp/fanout.py); a second
#                    concurrent job waits for admission.helper_lane()
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
BUDGET_MESSAGE = ("\n\nThinking budget reached. I will stop deliberating and "
                  "write the final answer now.\n")


def _shares() -> dict:
    try:
        import budget as _budget
        b = _budget.budgets()
        return {"main": b["main"], "helper": b["helper"], "pool": b["pool"]}
    except Exception:                                            # noqa: BLE001
        return {"main": 102400, "helper": 61440, "pool": 163840}


def estimate_prompt_tokens(body: dict) -> int:
    """A deliberately HIGH estimate of the prompt's tokens, from its size.

    ~3 characters per token over messages and tools. Over-estimating only
    lowers the thinking room a little; under-estimating could overflow the
    share. The exact count would cost a /tokenize round trip per request.
    """
    import json as _json
    n = 0
    for key in ("messages", "tools"):
        v = body.get(key)
        if v:
            try:
                n += len(_json.dumps(v, ensure_ascii=False))
            except (TypeError, ValueError):
                n += len(str(v))
    return n // 3


def budget(client_max_tokens, thinks: bool = True, cap: int | None = None,
           role: str = "main", share_n: int = 1,
           prompt_tokens: int = 0) -> dict:
    """The upstream token fields for one request, derived from its share.

    `cap` overrides the derived thinking room for ONE request -- the
    benchmark sets it through X-Yamadori-Features {"reasoning_cap": N} to
    measure whether longer thinking helps. Clamped to [MIN_THINKING, pool].
    """
    try:
        answer = int(client_max_tokens or 0)
    except (TypeError, ValueError):
        answer = 0
    answer = max(answer, A_MIN)
    if not thinks:
        return {"max_tokens": answer}
    s = _shares()
    window = s["helper" if role == "helper" else "main"] // max(int(share_n), 1)
    thinking = max(window - int(prompt_tokens or 0) - answer, MIN_THINKING)
    if cap is not None:
        try:
            thinking = min(max(int(cap), MIN_THINKING), s["pool"])
        except (TypeError, ValueError):
            pass
    return {"max_tokens": thinking + answer,
            "reasoning_budget_tokens": thinking,
            "reasoning_budget_message": BUDGET_MESSAGE}


# THE COMPACTION BUDGET (operator, 2026-09-24). A client's compaction
# (selection.utility_kind == "compaction") is served on the conversation it
# summarises (mcp/compaction.py) and THINKS at that conversation's own effort
# -- at every effort, medium included -- with the conversation's own
# sampling; thinking is off only where the conversation itself runs with it
# off (compaction.prefix_fields; interim defaults of 2026-09-24 until
# docs/COMPACTION-RESEARCH.md's eval runs). Its thinking budget is
# COMPACTION_THINKING, 2,048: from docs/CONSTRAINTS.md, 5 of the 7
# self-finished thinking runs fit in 2,048, on coding prompts, n=1 each -- NOT
# a compaction measurement. Its answer allowance is at least
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
COMPACTION_THINKING = int(os.environ.get("YAMADORI_COMPACTION_THINKING",
                                         "2048"))


def compaction_budget(client_max_tokens, prompt_tokens: int,
                      helper_active: int = 0,
                      shares: dict | None = None) -> dict:
    """The answer allowance for one compaction, and the record of why:
    {budget, cap, client_max_tokens, answer, prompt_estimate, main_share,
    pool, helper_active, window, room, fits, draws_on_spare}."""
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
    window = s["pool"] - (s["helper"] if active else 0)
    rec = {"budget": b, "cap": cap, "client_max_tokens": client or None,
           "prompt_estimate": prompt, "main_share": s["main"],
           "pool": s["pool"], "helper_active": active, "window": window}
    if prompt + answer <= window:
        rec.update(answer=answer, fits=True,
                   room=("pool: no second brain running, its share is spare"
                         if not active else
                         "pool less the running second brain's share"),
                   draws_on_spare=prompt + answer > s["main"])
    else:
        rec.update(answer=max(min(answer, s["main"] - prompt), A_MIN),
                   fits=False, draws_on_spare=False,
                   room=(f"main share: prompt ~{prompt} + answer {answer} "
                         f"exceed the {window}-token window"
                         + (" with a second brain running" if active else "")))
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


def enforce_sampling(body: dict, thinks: bool = True) -> tuple[dict, dict]:
    """(sampling fields to send, the record for x_yamadori.sampling).

    The record is {enforced: {...}, client_overridden: {key: client value}}
    -- only keys the caller set to something else appear in the second."""
    want = dict(VENDOR_SAMPLING if thinks else VENDOR_SAMPLING_INSTRUCT)
    over = {k: body[k] for k in want
            if k in body and body[k] is not None and body[k] != want[k]}
    return want, {"enforced": dict(want), "client_overridden": over}


def apply(body: dict, tier: dict, role: str = "main",
          share_n: int = 1) -> dict:
    """Write the tier's sampling settings into the upstream request.

    The augmentation flags are NOT written here -- they are read by the proxy,
    which decides what to run. Only what the model server itself understands
    is set: the effort, whether it thinks, and the token budget, derived from
    the request's share of the pool (role, share_n) and its own prompt.
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
    fields, record = enforce_sampling(body, bool(tier["thinks"]))
    out.update(fields)
    out["_sampling"] = record
    out.update(budget(client_max, tier["thinks"],
                      tier.get("reasoning_cap"), role=role, share_n=share_n,
                      prompt_tokens=estimate_prompt_tokens(body)))
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
        out["reasoning_effort"] = safe_effort(tier["effort"])
        out["enable_thinking"] = True
    else:
        out["enable_thinking"] = False
        out.pop("reasoning_effort", None)
        out.pop("reasoning_budget_tokens", None)
        out.pop("reasoning_budget_message", None)
    return out


def rebudget(payload: dict, role: str = "main", share_n: int = 1) -> dict:
    """Re-derive an ALREADY-SHAPED payload's budget for a different share:
    a fan-out candidate written by the second brain gets the helper's share
    (role="helper", share_n=1), and its prompt now includes whatever it was
    given -- the tie-breaker's carries both candidates. share_n > 1 splits a
    share n ways, for n concurrent samples; nothing in the proxy runs those
    since fan-out went sequential (2026-09-23)."""
    if not payload.get("enable_thinking", True) or \
            "reasoning_budget_tokens" not in payload or \
            payload.get("_fixed_budget"):
        # _fixed_budget: a compaction keeps the smallest thinking budget it
        # was given (proxy._serve_compaction), not the share's.
        return dict(payload)
    out = dict(payload)
    answer = max(int(out.get("max_tokens") or 0)
                 - int(out.get("reasoning_budget_tokens") or 0), A_MIN)
    out.update(budget(answer, True, role=role, share_n=share_n,
                      prompt_tokens=estimate_prompt_tokens(out)))
    return out


def describe() -> str:
    lines = ["reasoning_effort selects how hard this tries:"]
    for n in ORDER:
        t = TIERS[n]
        bits = []
        if t["retrieval"]:
            bits.append("search")
        if t["hints"]:
            bits.append("skills")
        if t["fanout"] > 1:
            bits.append(f"fan-out x{t['fanout']}")
        if t["investigate"]:
            bits.append("deep thinking")
        if t.get("check_code"):
            bits.append("check_code")
        lines.append(f"  {n:<8} effort {t['effort']:<7} "
                     f"{', '.join(bits) or 'nothing'} -- {t['why']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print()
    for v in ("xhigh", "HIGH", "none", None, "nonsense", "med"):
        t = resolve({"reasoning_effort": v})
        print(f"  {str(v):<10} -> {t['name']:<8} "
              f"effort={t['effort']:<7} max_tokens={apply({}, t)['max_tokens']:<6} "
              f"fanout={t['fanout']} "
              f"investigate={t['investigate']}")
    t = resolve({"reasoning_effort": "high"}, overrides={"fanout": 1})
    # `budget` was renamed to `floor` and this line was not updated, so the
    # self-test raised KeyError instead of printing -- which is why nobody
    # noticed. A self-test that crashes is worse than none: it looks like
    # coverage and provides none.
    print(f"\n  high with fanout forced to 1 (factorial arm): "
          f"fanout={t['fanout']}, "
          f"overridden={t.get('overridden')}")


# A true A/B needs the augmentations toggled at a FIXED reasoning level.
# The presets deliberately move both at once, which is right for a product
# dial and useless for attribution: if `max` beats `minimal`, the bundle won
# and nobody can say which half did it.
#
# These are the two arms that isolate it. Same effort, same floor, same
# template, same preamble -- only the injected tools and hints differ.
AB_ARMS = {
    "aug_off": {"retrieval": False, "hints": False, "fanout": 1,
                "investigate": False},
    "aug_on": {"retrieval": True, "hints": True, "fanout": 1,
               "investigate": False},
}


def ab(effort: str = "medium", arm: str = "aug_off") -> dict:
    """One arm of the augmentation A/B, at a reasoning level you choose.

    `fanout` stays at 1 in BOTH arms on purpose. Fan-out multiplies generation
    cost and would dominate any difference, so it gets its own experiment
    rather than riding along inside this one.
    """
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
    # `delegate` offers `delegate_investigation` as a TOOL, the benchmark arm
    # kept behind a flag (proxy.DELEGATE_TOOL). Every flag set here is FORCED
    # on or off; flags left out are ALLOWED by the tier and decided per request
    # by mcp/selection.py -- `tier["overridden"]` is how it tells them apart.
    # `utility` forces the client-utility-call decision (proxy.utility_of,
    # selection.utility_call) either way: true -> the bare model at minimal,
    # false -> a task turn whatever the request looks like.
    types_ok = {"retrieval": bool, "hints": bool, "investigate": bool,
                "fanout": int, "effort": str, "delegate": bool,
                "reasoning_cap": int, "check_code": bool, "repair": bool,
                "utility": bool}
    out = {k: v for k, v in d.items()
           if k in types_ok and isinstance(v, types_ok[k])
           and not (types_ok[k] is int and isinstance(v, bool))}
    return out or None
