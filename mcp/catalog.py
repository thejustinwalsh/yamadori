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
    "yamadori-max": ("bonsai", "max"),
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


def public_list() -> dict:
    """The OpenAI `/v1/models` payload, product names only."""
    created = int(time.time())
    names = list(CATALOG) if EXPOSE_INTERNAL else list(PUBLIC)
    data = [{"id": name, "object": "model", "created": created,
             "owned_by": OWNER}
            for name in names]
    return {"object": "list", "data": data}


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
