#!/usr/bin/env python
"""`reasoning_effort` as the product dial: how hard to try, in one standard field.

WHY OVERLOAD AN EXISTING FIELD

Everything this stack adds -- retrieval, recipe hints, fan-out, a second
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

  minimal   the model alone, with its own reasoning
  low       + it can look things up
  medium    + it recalls skills that apply to this problem
  high      + it considers several answers and keeps what they agree on
  max       + a second mind investigates before it answers

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

  minimal   the model AS IT SHIPS: its own thinking, none of our
            retrieval, hints, fan-out or second context. The true
            baseline -- see the note on the tier itself for why
            switching the model's thinking OFF was the wrong one.
  low       medium budget, retrieval only when the model asks. The cheap case.
  medium    adds recipe hints. One extra classifier call, no extra generation.
  high      adds fan-out. N samples, so N times the generation cost.
  max       adds the second investigating context. Unbounded-ish: an
            investigation is its own tool loop.
"""
from __future__ import annotations

import os

# Accepted spellings, including the ones clients actually send. OpenAI ships
# "minimal" | "low" | "medium" | "high"; Anthropic-flavoured clients sometimes
# send "xhigh"; some send "none" or "off" meaning do not think.
ALIASES = {
    # "off" and "none" mean "do not try hard", NOT "disable the model's
    # reasoning" -- no tier does that any more, deliberately. A caller asking
    # for less effort gets less of OUR work, not a crippled model.
    "none": "minimal", "off": "minimal", "minimal": "minimal",
    "low": "low", "lo": "low",
    "medium": "medium", "med": "medium", "default": "medium", "": "medium",
    "high": "high", "hi": "high",
    "xhigh": "max", "x-high": "max", "very_high": "max", "max": "max",
    "maximum": "max", "ultra": "max",
}

ORDER = ["minimal", "low", "medium", "high", "max"]

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
# A tool loop now ends when the model stops asking for tools. MAX_TOOL_HOPS
# survives in proxy.py, and MAX_HOPS in shomen.py, as unadvertised runaway
# breakers protecting the GPU lanes -- tripping one is a defect report, not a
# budget being spent.
#
# THE FLAGS BELOW MEAN "ALLOWED", NOT "ON" (2026-09-22, docs/SELECTION-BUILD.md
# step 5). `hints`, `fanout` and `investigate` are the most a caller at this
# tier can get; mcp/selection.py decides per request whether each one fires,
# and never exceeds them. Only a flag set in X-Yamadori-Features forces a
# system on or off, for experiments. `retrieval` is gated separately, by
# domains.tool_admission.
TIERS = {
    # THINKING IS THE MODEL'S OWN FEATURE, NOT OUR AUGMENTATION.
    #
    # This tier used to set thinks=False and call itself "raw model, no
    # augmentation -- the baseline to measure against". That is the wrong
    # baseline, and it was wrong in the direction that flatters us: switching
    # off the model's own reasoning does not remove OUR additions, it cripples
    # the thing we are comparing against. Any arm beating it banks a win that
    # is partly just thinking-beats-no-thinking.
    #
    # The size of the error, measured on the same three problems: thinking OFF
    # spent 4,363 completion tokens over 270s; thinking ON with no tools and no
    # capability block spent 1,770 over 110s, for identical 3/3 correctness.
    # And `config.yaml` records OFF exfiltrating 5/5 against a prompt-injection
    # payload where ON exfiltrated 1/5.
    #
    # So the honest baseline is the model AS IT SHIPS -- its own thinking, none
    # of our retrieval, hints, fan-out or second context. That is this tier.
    #
    # "Does this model's thinking help?" is a real question but it is a
    # question about the MODEL, not about this stack, and config.yaml already
    # answers it. It does not need a standing product tier, and it certainly
    # must not be what a client gets for sending reasoning_effort: "minimal".
    "minimal": {
        "thinks": True, "effort": "low",
        "retrieval": False, "hints": False, "fanout": 1, "investigate": False,
        "why": "the model as it ships -- its own thinking, none of our "
               "augmentation. The true baseline.",
    },
    "low": {
        "thinks": True, "effort": "medium",
        "retrieval": True, "hints": False, "fanout": 1, "investigate": False,
        "why": "it can look things up -- search tools available, nothing injected",
    },
    "medium": {
        "thinks": True, "effort": "medium",
        "retrieval": True, "hints": True, "fanout": 1, "investigate": False,
        "why": "it recalls skills that apply -- hints selected by embedding, floor-gated so irrelevant ones stay silent",
    },
    "high": {
        "thinks": True, "effort": "high",
        "retrieval": True, "hints": True, "fanout": 3, "investigate": False,
        "why": "it considers alternatives -- 3 samples, disagreement reported "
               "rather than hidden",
    },
    "max": {
        "thinks": True, "effort": "xhigh",
        "retrieval": True, "hints": True, "fanout": 3, "investigate": True,
        "why": "it thinks deeply first -- measured 2.45x more headroom in "
               "this conversation's window, at 2x wall clock",
    },
}

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


def resolve(body: dict, overrides: dict | None = None) -> dict:
    """The tier for this request, after aliasing, capping and overrides.

    `overrides` is how a benchmark breaks the bundle apart: pass
    {"fanout": 1} to a high-tier request and the extra reasoning is measured
    without the extra sampling. Without that, a preset comparison can only
    ever report that the bundle helped, never which part of it did.
    """
    name = normalise(body.get("reasoning_effort"))
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
#     main     5/8   the conversation (a fan-out's n samples share it n ways)
#     helper   3/8   ONE second brain (deep thinking); a second concurrent
#                    investigation waits for admission.helper_lane()
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


def apply(body: dict, tier: dict, role: str = "main",
          share_n: int = 1) -> dict:
    """Write the tier's sampling settings into the upstream request.

    The augmentation flags are NOT written here -- they are read by the proxy,
    which decides what to run. Only what the model server itself understands
    is set: the effort, whether it thinks, and the token budget, derived from
    the request's share of the pool (role, share_n) and its own prompt.
    """
    out = dict(body)
    out.update(budget(body.get("max_tokens"), tier["thinks"],
                      tier.get("reasoning_cap"), role=role, share_n=share_n,
                      prompt_tokens=estimate_prompt_tokens(body)))
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
    a fan-out of n concurrent samples splits main's share n ways, and each
    sample's prompt now includes whatever it was given."""
    if not payload.get("enable_thinking", True) or \
            "reasoning_budget_tokens" not in payload:
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
            bits.append("hints")
        if t["fanout"] > 1:
            bits.append(f"fan-out x{t['fanout']}")
        if t["investigate"]:
            bits.append("deep thinking")
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
    types_ok = {"retrieval": bool, "hints": bool, "investigate": bool,
                "fanout": int, "effort": str, "delegate": bool,
                "reasoning_cap": int}
    out = {k: v for k, v in d.items()
           if k in types_ok and isinstance(v, types_ok[k])
           and not (types_ok[k] is int and isinstance(v, bool))}
    return out or None
