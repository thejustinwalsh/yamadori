#!/usr/bin/env python
"""Who gets how much of the shared context pool.

THE THING THAT MAKES THIS NECESSARY

llama-server runs four slots over ONE unified KV cache. Verified rather than
assumed: four slots each report n_ctx 131072, which would be 524,288 tokens of
KV and about 22 GB at q8_0, while the measured footprint is 5.8 GB. So the
per-slot figure is a ceiling, not an allocation, and the pool is the real
number: 147,456, the value config.yaml launches with at `-c`.

Unified KV has no per-slot reservation. Whoever asks first gets the tokens. A
long conversation and three helper investigations draw from the same pool, and
nothing in the server stops a helper from crowding out the conversation that
spawned it.

So the split has to be enforced one layer up, here, by capping what each KIND
of request is allowed to send. The proxy is the only component that knows
which requests are the user's and which are its own.

THE PROPORTIONS

Expressed as fractions of whatever the server actually has, discovered at
startup, so raising `-c` raises both budgets and nothing needs editing twice.

  main      5/8 of the pool -- the conversation
  helper    3/8 of the pool, for ONE second brain (deep thinking)

At a pool of 147,456 that is exactly:

  main       92,160
  helper     55,296  x 1
             -------
             147,456  (the pool, with nothing left unclaimed)

and at the 163,840 config.yaml launches with now: 102,400 + 61,440.

ONE SECOND BRAIN, NOT TWO, 2026-09-22 (later the same day). The 1/2 + 2 x 1/4
split held a quarter of the pool for a second concurrent investigation. Within
one conversation that never happens: delegate_investigation and the
selection-triggered investigation both run under admission.helper_lane(), and
a conversation's tool calls run one at a time. Two helpers at once needed two
separate conversations each investigating at the same moment -- rare on a
one-operator stack. The quarter it held was not VRAM (unified KV allocates
cells on demand) but it WAS a cap: the conversation could never grow past half
the pool, and the one investigator past a quarter, to leave room for a third
context that did not exist. The operator's call: give it back to the two that
run. A second concurrent investigation now waits for the lane
(admission.HELPER_LANES = 1) and is refused with HELPER_BUSY if it cannot get
it, which the model is told in so many words.

THE EARLIER SPLIT, 2026-09-22 (superseded above). The code had drifted to main 60%,
ONE helper at 25%, and ~15% "reserve" justified as prompt scratch space. That
justification does not hold: llama-server allocates its prefill compute buffer
as separate VRAM at load, so scratch never comes out of the KV pool's token
cells, and a reserve of cells bought nothing. The decided split is half for the
conversation and a quarter for each of up to two second brains. The
reserve argument still holds for the current split: reserve is 0.

EVERY TOKEN BUDGET DERIVES FROM THIS. tiers.budget() gives a request its
role's share minus its prompt and its answer allowance as THINKING room; a
fan-out of n samples shares main's 5/8 n ways. There is no free-floating
thinking number.

WHAT IT COSTS TO RAISE THE POOL

Measured on this hardware, KV at q8_0 runs about 44 KiB/token. Against 6.45 GB
of overhead (5.95 GB of weights plus roughly 0.5 GB of compute buffers) on a
16.3 GB card, which is the same arithmetic `what_if()` prints:

  147,456 (today)   6.19 GiB KV   total ~12.6 GB   ~3.7 GB spare on GPU0
  196,608           8.25 GiB KV   total ~14.7 GB   ~1.6 GB spare
  262,144          11.00 GiB KV   total ~17.5 GB   DOES NOT FIT in 16.3 GB

196,608 is the largest that fits on paper, and it was tried and BACKED OUT:
~1.6 GB spare is not room for the prefill scratch that a long prompt actually
needs, and `arc193_a` died on all three benchmark arms there. 147,456 is what
ships. 256k needs the model split across both GPUs, which costs cross-device
KV traffic on every token.

Note that these splits are NOT "128k for the conversation, 64k for helpers".
At 5/8 + 3/8 the conversation gets 102,400 tokens out of 163,840 -- short of
128k. (Under the retired 1/2 + 2 x 1/4 split it was 81,920; under 60/25/15,
98,304.) Any doc that
promises 128k to a single conversation is quoting the old per-slot ceiling,
not a budget.

WHAT THIS DOES NOT DO

Raise the pool. That is a launch flag on llama-server and a restart. This
module only decides how a pool of a given size is shared, and reports what a
different size would buy.
"""
from __future__ import annotations

import json
import os
import urllib.request

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
DIRECT = os.environ.get("YAMADORI_MODEL_SERVER", "http://127.0.0.1:10001")

# Fractions of the discovered pool: 5/8 + 1 x 3/8 = the whole pool.
MAIN_SHARE = float(os.environ.get("YAMADORI_MAIN_SHARE", "0.625"))
HELPER_SHARE = float(os.environ.get("YAMADORI_HELPER_SHARE", "0.375"))
HELPERS = int(os.environ.get("YAMADORI_HELPERS", "1"))
# The conversation never gets less than this share however the fractions are
# set, because a helper starving the thing it was spawned to serve is the one
# outcome that makes the whole arrangement worse than not having it.
MAIN_FLOOR = float(os.environ.get("YAMADORI_MAIN_FLOOR", "0.50"))
# Measured on this hardware at --cache-type-k q8_0 --cache-type-v q8_0.
KV_KIB_PER_TOKEN = float(os.environ.get("YAMADORI_KV_KIB", "44"))

_POOL: int | None = None


def pool_size(refresh: bool = False) -> int:
    """Total context the server was launched with, asked rather than assumed.

    Falls back to 131072 only if the server cannot be reached, and says so in
    the report, because a silently wrong pool size would quietly mis-budget
    every request.

    FLAGGED, NOT FIXED: that 131072 fallback is stale -- config.yaml has
    shipped `-c 147456` since the 196608 experiment was backed out, so an
    unreachable server now under-budgets by 16,384 tokens instead of matching
    the live pool. Changing it is a behaviour change, not a comment fix, so it
    is left to a deliberate edit. mcp/dash_vitals.py already surfaces the
    ambiguity in the UI (it notes that a pool of exactly 131072 may be this
    fallback rather than a measurement).
    """
    global _POOL
    if _POOL is not None and not refresh:
        return _POOL
    for url in (f"{DIRECT}/props", f"{UPSTREAM}/props"):
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                d = json.load(r)
            n = ((d.get("default_generation_settings") or {}).get("n_ctx")
                 or d.get("n_ctx"))
            if n:
                _POOL = int(n)
                return _POOL
        except Exception:                                        # noqa: BLE001
            continue
    _POOL = 131072
    return _POOL


def budgets(pool: int | None = None) -> dict:
    p = pool or pool_size()
    main = max(int(p * MAIN_SHARE), int(p * MAIN_FLOOR))
    helper = int(p * HELPER_SHARE)
    if main + HELPERS * helper > p:
        helper = max((p - main) // max(HELPERS, 1), 0)
    return {"pool": p, "main": main, "helper": helper, "helpers": HELPERS,
            "reserve": p - main - HELPERS * helper,
            "gib": round(p * KV_KIB_PER_TOKEN / 1024 / 1024, 2)}


def cap_for(kind: str) -> int:
    """Token ceiling for one kind of request: 'main' or 'helper'.

    Anything else gets the main budget. It used to be `b.get(kind, main)`,
    which answered `cap_for("pool")` with the WHOLE pool and `cap_for("gib")`
    with a float -- every key of the report was a valid "kind".
    """
    b = budgets()
    return b["helper"] if kind == "helper" else b["main"]


def fits(text_tokens: int, kind: str = "main") -> tuple[bool, str]:
    cap = cap_for(kind)
    if text_tokens <= cap:
        return True, ""
    return False, (f"{text_tokens} tokens exceeds the {kind} budget of {cap} "
                   f"(pool {pool_size()}). Trim the input or raise -c.")


def what_if(sizes=(131072, 147456, 163840, 196608, 262144)) -> str:
    """What raising the pool would cost and buy, in one table.

    The default list includes 147456, the pool config.yaml ships at `-c`, so
    the row actually running is in the table next to the candidates. It was
    missing until 2026-09-22 (asserted in mcp/test_budget.py).
    """
    lines = [f"  {'pool':>8}{'KV GiB':>9}{'main':>9}{'helper':>9}{'reserve':>9}  fits 16.3 GB card?"]
    # 5.95 GB of weights plus roughly 0.5 GB of compute buffers.
    overhead = 5.95 + 0.5
    for p in sizes:
        b = budgets(p)
        total = overhead + b["gib"]
        ok = "yes" if total < 16.3 else "NO"
        lines.append(f"  {p:>8}{b['gib']:>9.2f}{b['main']:>9}{b['helper']:>9}"
                     f"{b['reserve']:>9}  {ok} ({total:.1f} GB total)")
    return "\n".join(lines)


if __name__ == "__main__":
    p = pool_size()
    b = budgets()
    print(f"  server reports a pool of {p} tokens "
          f"({b['gib']} GiB of KV at {KV_KIB_PER_TOKEN} KiB/token)")
    print(f"    main   {b['main']:>7}  ({MAIN_SHARE:.0%}, floor {MAIN_FLOOR:.0%})")
    print(f"    helper {b['helper']:>7}  ({HELPER_SHARE:.0%} each, x{b['helpers']})")
    print(f"    reserve{b['reserve']:>7}  (whatever the shares leave unclaimed)")
    print()
    print(what_if())
    print("\n  Unified KV has no per-slot reservation, so these are enforced")
    print("  HERE, by capping what each kind of request may send. The server")
    print("  would happily let a helper consume the conversation's tokens.")
