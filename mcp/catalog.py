#!/usr/bin/env python
"""The public model list. One product name, internals never advertised.

WHY THE INTERNAL NAMES MUST NOT LEAK

Until now `/v1/models` was proxied straight through, so a client saw `bonsai`,
`bonsai-vision`, `critic-disabled`, `embeddings` and `reranker`. Three things
are wrong with that.

It advertises a bypass. Anything a client can name, it can request, and a
request naming `bonsai` is a request that skipped the retrieval, the tier
resolution, the secret filtering and the audit log. The proxy stops being the
way in and becomes one of two ways in.

It advertises a footgun. `embeddings` and `reranker` are in the list, they are
not chat models, and a client that picks one gets a confusing failure rather
than a clear refusal.

It advertises implementation. `Ternary-Bonsai-2-27B-Abliterated-PTQ1_0` is a
build decision. Swapping it should not be a breaking change for every caller,
and it will be swapped.

THE MODEL NAME AS A SECOND TIER DIAL

`reasoning_effort` is the intended dial, but plenty of clients expose a model
picker and nothing else -- no effort field, no custom headers. For those the
only control surface is the name, so the tiers RESOLVE as model variants:
asking for `yamadori-fast` does what `reasoning_effort: "low"` would have.

They are not advertised. One name is the product; a list of five is a menu of
implementation details that callers would then depend on. A benchmark that
wants to drive tiers by name sets YAMADORI_EXPOSE_INTERNAL=1.

An explicit `reasoning_effort` in the body still wins, because a caller who
went to the trouble of setting it meant it.

NAMES ARE CLAIMED, NOT ASSIGNED

A product name says the thing behind it is ours. `metsumi` is reserved for a
fine-tune of the decision model that does not exist yet; until it does, the
decision layer runs stock weights and stays unnamed. See RESERVED below.
"""
from __future__ import annotations

import os
import time

# WHAT IS ADVERTISED, AND WHAT MERELY WORKS.
#
# One name. `yamadori` carries no tier of its own: it takes whatever the
# request asks for, or the default.
#
# Everything else is an implementation detail. Tier variants, the vision
# model, the embedding and rerank endpoints all still RESOLVE if a caller
# names them, because breaking a working integration to tidy a list is a bad
# trade -- but they are not offered, so no model picker can present them and
# no one can come to depend on a name that is really a build decision.
#
# Set YAMADORI_EXPOSE_INTERNAL=1 to list them all, which is what a benchmark
# comparing tiers by model name needs.
PUBLIC = {
    "yamadori": ("bonsai", None),
}

# RESERVED, NOT SHIPPED.
#
# `metsumi` is the name for OUR tuned decision model, and the decision model
# running today is stock `convaiinnovations/laya`. Publishing someone else's
# weights under this product's name would be a claim the repo has not earned,
# and it is the kind of claim everything else here is careful not to make.
#
# It is also not a working endpoint: Laya speaks its own /decide protocol on
# port 1237, not chat completions, so a caller selecting it would get a
# confusing failure rather than a classifier.
#
# The name goes live when there is a fine-tune to attach it to -- the pairs
# exist (bench/mechanisms/leetcode_features.json, 2,641 labelled problems)
# and the training does not. Until then the decision layer is internal and
# unnamed, which is accurate.
RESERVED = {
    "metsumi": "reserved for a fine-tune of the decision model. Today that "
               "layer runs stock Laya, which is not ours to rename",
}

# Resolvable, deliberately unadvertised.
INTERNAL = {
    "yamadori-fast": ("bonsai", "low"),
    "yamadori-xhigh": ("bonsai", "xhigh"),
    "yamadori-max": ("bonsai", "max"),
    # the A4000 vision copy again since layout v2 (2026-09-29; the main model
    # saw from 2026-09-27 until then)
    "yamadori-vision": ("bonsai-vision", None),
    "yamadori-embed": ("embeddings", None),
    "yamadori-rerank": ("reranker", None),
}

CATALOG = {**PUBLIC, **INTERNAL}
EXPOSE_INTERNAL = os.environ.get("YAMADORI_EXPOSE_INTERNAL") == "1"

OWNER = os.environ.get("YAMADORI_OWNER", "yamadori")
# One alias for the old name, so the benchmark runs and any client already
# pointed at `bonsai` keep working while the rename settles. Deliberately not
# advertised: it exists to avoid breaking callers, not to invite new ones.
LEGACY = {"bonsai": ("bonsai", None), "bonsai-agent": ("bonsai", None)}


# IMAGE MODELS, by the name a client may send in `model` on
# POST /v1/images/generations. Kept apart from CATALOG on purpose: a chat
# request naming `yamadori-image` must not resolve to an image server, and an
# image request naming `yamadori` must not pick a chat model. The value is the
# image-model id in mcp/images.py MODELS. Not advertised in /v1/models, which
# lists chat models; GET /dash/api/settings/image lists these.
IMAGE_MODELS = {
    "yamadori-image": "base",
    "yamadori-image-turbo": "turbo",
}


def resolve_image(name) -> str | None:
    """The image-model id a request names, or None when it names none of ours.

    None means "use the caller's preference". OpenAI SDKs fill `model` with
    their own default (`dall-e-3`, `gpt-image-1`); that is not a choice, so it
    is ignored exactly as it was before there were two image models.
    """
    if not isinstance(name, str):
        return None
    return IMAGE_MODELS.get(name.strip())


def context_window(model: str | None = None) -> int:
    """The conversation window a client may plan for: the MAIN share
    (mcp/budget.py), read at runtime. Under the cap layout (operator,
    2026-09-28) it is the main CAP -- the measured VRAM line of the tiered KV
    pool -- and every conversation is advertised all of it; under the split
    it is the pool less the second brain's share.

    Not the pool: the rest is the child's and a second conversation's. It is THE LIMIT ENFORCED
    (2026-09-25, docs/OPENAI-CONFORMANCE.md C1): a request whose prompt,
    counted by the model server, leaves less than the generation floor in
    this window is refused with 400 context_length_exceeded
    (proxy.check_client_prompt, proxy.window_limit reads the same share) --
    it is no longer landed. Not the per-slot n_ctx either, which is a
    ceiling four slots share. A harness reads this to decide when to
    compact; advertising nothing left Hermes to guess."""
    import budget
    # the per-model window only when the table gives this model one (budget.model_window); otherwise the call is the
    # one it always was, budget.budgets() -- callers and suites that stand in for budgets(pool=None) keep working
    if model and budget.model_window(model):
        return int(budget.budgets(model=model)["main"])
    return int(budget.budgets()["main"])


def tier_window(tier: str | None) -> int:
    """ONE MODEL PER EFFORT TIER (mcp/tier_models.py): the window of the model that serves `tier` (the default tier
    when None). The same number proxy.window_limit enforces for a request at that tier."""
    import tier_models
    import tiers
    return context_window(tier_models.model_for(tiers.normalise(tier or tiers.DEFAULT)))


# The field names clients read for a model's window: OpenRouter and most
# harnesses (Hermes among them) `context_length`; vLLM `max_model_len`; some
# UIs `context_window`. All three carry the same number.
CONTEXT_FIELDS = ("context_length", "max_model_len", "context_window")

# The names clients read for the output ceiling, at the top level of a model
# row (OpenRouter nests the same number under top_provider, which is also
# written). NOT `max_tokens`: Hermes Agent reads a top-level `max_tokens` as a
# last-resort CONTEXT length (agent/model_metadata.py
# _context_length_from_model_payload, L859-870 at 35b14ad), so a field by that
# name would be mistaken for the window by the one harness that looks hardest.
OUTPUT_FIELDS = ("max_completion_tokens", "max_output_tokens")

# THE OUTPUT CEILING WE ADVERTISE: 1/5 of the conversation window.
#
# What a client may send as max_tokens is not capped by the proxy: it is the
# ANSWER allowance (tiers.budget: answer = max(client, A_MIN)), and thinking
# gets the window less the prompt and that answer. So the true ceiling is
# the window less MIN_THINKING and the prompt -- a number that depends on
# the prompt, which no model list can state.
#
# Advertising that near-window figure would hurt twice. Harnesses send the
# advertised ceiling back as max_tokens, which would floor thinking at
# MIN_THINKING on every turn; and harnesses reserve it out of the window
# before compacting (input room = window - output), which would leave them
# no room at all. A fifth is the figure OpenRouter-style clients already
# derive when a provider states none (Roo-Code src/api/providers/fetchers/
# openrouter.ts L121-123 at b867ec9: `max_completion_tokens ||
# Math.ceil(context_length * 0.2)`), so stating it changes nothing for them
# and gives the rest a number. NOT MEASURED: no run here says a fifth is
# better than a quarter. It follows the pool (20,480 at -c 163840) and never
# drops below A_MIN, the smallest answer allowance the proxy sends.
OUTPUT_FRACTION = 5


def max_output(window: int | None = None) -> int:
    """The advertised output ceiling (see OUTPUT_FRACTION): the window / 5,
    never below tiers.A_MIN."""
    import tiers
    w = int(window if window is not None else context_window())
    return max(w // OUTPUT_FRACTION, tiers.A_MIN)


# Request fields the proxy honours, in OpenRouter's `supported_parameters`
# vocabulary where one exists. Only what is TRUE end to end:
#   max_tokens        the answer allowance (tiers.budget), thinking added
#   tools, tool_choice the client's own tools are merged with ours (proxy)
#   reasoning_effort  picks the tier (tiers.resolve; values in EFFORTS)
#   response_format, stop  passed to the model server unchanged
#   stream            SSE (server._stream)
# Left out on purpose: temperature, top_p, top_k, presence_penalty -- the
# vendor sampling is ENFORCED and a client's values are recorded as
# overridden (tiers.enforce_sampling); seed -- fan-out may deliver another
# candidate than the seeded one.
#   max_completion_tokens  read as max_tokens when that is absent, never
#                          passed upstream (tiers.apply)
#   reasoning              OpenRouter's object form; `reasoning.effort` picks
#                          the tier like reasoning_effort (tiers.resolve)
SUPPORTED_PARAMETERS = ("max_tokens", "max_completion_tokens", "tools",
                        "tool_choice", "reasoning", "reasoning_effort",
                        "response_format", "stop", "stream")


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_vision_cache: dict = {}


def swap_config() -> str:
    """llama-swap's config: YAMADORI_SWAP_CONFIG, else the repo's config.yaml
    (scripts/start-stack.bat starts llama-swap on %CD%\\config.yaml)."""
    return (os.environ.get("YAMADORI_SWAP_CONFIG") or "").strip() or \
        os.path.join(_ROOT, "config.yaml")


def vision_configured() -> bool:
    """Is a model that can see in llama-swap's config WITH its projector
    (`--mmproj` in its cmd): the vision copy (model.VISION_MODEL,
    `bonsai-vision` on the A4000 since layout v2), or the main model itself
    (the 2026-09-27 fold, which a config before the layout-v2 deploy still
    has)? Read from the file, cached by its mtime; never from the network.
    An unreadable config is False: the card never promises what cannot be
    checked. Kept out of mcp/vision.py, whose contract is that it opens no
    file (test_vision)."""
    path = swap_config()
    try:
        import model
        names = (model.VISION_MODEL, model.MODEL)
        key = (path, os.path.getmtime(path), names)
    except Exception:                                            # noqa: BLE001
        return False
    if key in _vision_cache:
        return _vision_cache[key]
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ok = any(isinstance(e, dict) and "--mmproj" in str(e.get("cmd") or "")
                 for e in ((cfg.get("models") or {}).get(n) for n in names))
    except Exception:                                            # noqa: BLE001
        ok = False
    _vision_cache.clear()
    _vision_cache[key] = ok
    return ok


def image_input() -> bool:
    """Do chat requests accept image parts? Yes while vision is AVAILABLE:
    on (YAMADORI_VISION is not 0; mcp/vision.py holds each image for this
    request and the text model reads it through yama_describe_image) AND the
    vision model configured with its projector (vision_configured). Read per
    call."""
    try:
        import vision
        return bool(vision.enabled()) and vision_configured()
    except Exception:                                            # noqa: BLE001
        return False


def _chat_card(window: int | None) -> dict:
    """The metadata of the chat product, computed now. Empty without a
    window: an unknown window is left out, never guessed, and every other
    number here derives from it."""
    import tiers
    if not window:
        return {}
    out = max_output(window)
    inputs = ["text", "image"] if image_input() else ["text"]
    desc = ("Local 27B coding model with a code-intelligence tool layer "
            "(TypeScript, Rust/WASM, three.js TSL, TypeGPU). The proxy runs "
            "its own tools; a client needs no tool configuration. "
            f"reasoning_effort {'/'.join(tiers.ORDER)} picks how hard it "
            f"works (default {tiers.DEFAULT}).")
    if "image" in inputs:
        desc += (" Images are accepted as base64 data URIs and read by a "
                 "vision pass; http(s) image URLs are not fetched.")
    card = {"name": "Yamadori", "description": desc}
    # Window fields first: Hermes takes the FIRST context key it meets in
    # the row's own order, nested dicts included (model_metadata.py
    # _extract_first_int, L458-461 at 35b14ad). Every one carries the same
    # number anyway.
    card.update({f: window for f in CONTEXT_FIELDS})
    card.update({f: out for f in OUTPUT_FIELDS})
    # Two conventions, the same lists (docs/VISION.md 4a): OpenRouter's and
    # llama.cpp router-mode's `architecture`, and models.dev's `modalities`,
    # which OpenCode's and Pi's catalog code map from. NO harness reads
    # either from a custom endpoint today (VISION.md 2a): configuration is
    # the switch (docs/VISION.md 4b); these say what is true.
    card["architecture"] = {
        "modality": "+".join(inputs) + "->text",
        "input_modalities": inputs,
        "output_modalities": ["text"],
    }
    card["modalities"] = {"input": list(inputs), "output": ["text"]}
    card["top_provider"] = {"context_length": window,
                            "max_completion_tokens": out}
    # Local, so free. OpenRouter-shaped clients compute cost from these.
    card["pricing"] = {"prompt": "0", "completion": "0"}
    card["supported_parameters"] = list(SUPPORTED_PARAMETERS)
    # Our extension, in the same `x_yamadori` block responses carry. No
    # key in it is named like a window or an output field, so a client
    # walking the row for one cannot pick up a number from here.
    card["x_yamadori"] = {
        "reasoning_effort": {"values": list(tiers.ORDER),
                             "default": tiers.DEFAULT,
                             "ceiling": tiers.normalise(tiers.CEILING)},
        "answer_allowance_floor": tiers.A_MIN,
    }
    try:
        import tier_models
        if tier_models.profiles_on():
            # ONE MODEL PER EFFORT TIER: each effort is served by its own model with its own window; the tokens a
            # conversation at that effort may occupy (the limit enforced). Model names stay internal.
            card["x_yamadori"]["reasoning_effort"]["tokens_by_value"] = {
                t: _window_or_none(t) for t in tiers.ORDER}
    except Exception:                                            # noqa: BLE001
        pass
    return card


def _is_chat(name: str) -> bool:
    """Does this name serve the chat model? The vision copy, embeddings and
    reranker have windows and parameters of their own, so the chat card
    would be false of them."""
    return CATALOG.get(name, LEGACY.get(name, (None,)))[0] == "bonsai"


def _row(name: str, created: int, window: int | None) -> dict:
    row = {"id": name, "object": "model", "created": created,
           "owned_by": OWNER}
    if _is_chat(name):
        # THE WINDOW IS PER MODEL (one model per effort tier): a name with a tier hint advertises the window of the
        # model that serves that tier; `yamadori` (no hint) the default tier's.
        hint = CATALOG.get(name, LEGACY.get(name, (None, None)))[1]
        if hint:
            window = _window_or_none(hint) or window
        row.update(_chat_card(window))
    return row


def _window_or_none(tier: str | None = None) -> int | None:
    try:
        return tier_window(tier)
    except Exception:                                            # noqa: BLE001
        return None


def public_list() -> dict:
    """The OpenAI `/v1/models` payload, product names only, each chat model
    with its full card (`_chat_card`): window, output ceiling, modalities,
    supported parameters -- enough that a harness needs no configuration."""
    created = int(time.time())
    names = list(CATALOG) if EXPOSE_INTERNAL else list(PUBLIC)
    window = _window_or_none()
    return {"object": "list",
            "data": [_row(name, created, window) for name in names]}


def model_card(name: str | None) -> dict:
    """`GET /v1/models/{id}`: one row, never a 404.

    An advertised name gets its own row. Anything else gets the product's
    row: an unknown name, because that is what a chat request naming it
    reaches (`resolve`); an internal name not exposed, because a distinct
    answer -- a 404 or a card of its own -- would advertise that it exists."""
    key = (name or "").strip()
    advertised = CATALOG if EXPOSE_INTERNAL else PUBLIC
    return _row(key if key in advertised else "yamadori",
                int(time.time()), _window_or_none())


def resolve(name: str | None) -> tuple[str | None, str | None, bool]:
    """(internal model, tier hint, is_known) for a requested model name.

    An unknown name resolves to the default product rather than erroring.
    A client that sends its own default string -- "gpt-4", "default", an empty
    field -- should get a working answer, not a 404 about a model catalogue it
    never asked to learn.
    """
    key = (name or "").strip()
    if key in CATALOG:
        internal, tier = CATALOG[key]
        return internal, tier, True
    if key in LEGACY:
        internal, tier = LEGACY[key]
        return internal, tier, True
    internal, tier = PUBLIC["yamadori"]
    return internal, tier, False


def rewrite_response(payload: dict, public_name: str) -> dict:
    """Report the product name back, not the internal one.

    The model field is echoed by clients into logs, UIs and transcripts. If it
    says `bonsai` the internal name leaks anyway, one layer later, and a user
    reading their own transcript learns a name they cannot use.
    """
    if isinstance(payload, dict) and payload.get("model"):
        payload["model"] = public_name
    # This is also the one place every non-streamed reply passes through, so
    # it is where the proxy's own bookkeeping comes off. `_transport` carries
    # timings that are ours to log, not the caller's to parse, and an
    # underscored field in an OpenAI-shaped body is a schema violation waiting
    # to confuse a strict client.
    if isinstance(payload, dict):
        for k in [k for k in payload if isinstance(k, str) and k.startswith("_")]:
            payload.pop(k, None)
    return payload


def describe() -> str:
    def row(name, pair):
        internal, tier = pair
        t = f"tier {tier}" if tier else "tier from the request"
        return f"  {name:<18} -> {internal:<14} {t}"

    lines = ["advertised:"]
    lines += [row(n, p) for n, p in PUBLIC.items()]
    for n, why in RESERVED.items():
        lines.append(f"  {n:<18} RESERVED -- {why}")
    lines.append("resolvable, not advertised"
                 + (" (EXPOSED: YAMADORI_EXPOSE_INTERNAL=1)"
                    if EXPOSE_INTERNAL else "") + ":")
    lines += [row(n, p) for n, p in INTERNAL.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    print(describe())
    print()
    for n in ("yamadori", "yamadori-max", "bonsai", "gpt-4", "", None):
        internal, tier, known = resolve(n)
        print(f"  {str(n)!r:<16} -> {internal:<14} tier={tier}  "
              f"{'known' if known else 'unknown, defaulted'}")
    print()
    print(json.dumps(public_list(), indent=2)[:300])
