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

  main      0.70 of the pool -- the conversation (5/8 until 2026-09-25)
  helper    0.30 of the pool, for ONE second brain (deep thinking; 3/8 until 2026-09-25)

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
fan-out's extra candidates are written one at a time by the second brain,
each with the helper's whole 3/8 (mcp/fanout.py, 2026-09-23; until then n
concurrent samples split main's 5/8 n ways). There is no free-floating
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

THE CAP LAYOUT (operator, 2026-09-28; replaces the split above when a cap is
known). "Always run from VRAM." The main model's KV pool is tiered
(--kv-vram-cells N: cells [0, N) in VRAM, the rest in pinned host RAM, read
over PCIe on every step once a sequence reaches past them -- pagoda-h6 decoded
6-15 tok/s from there). So nothing is sized as a fraction of the pool:

  main      MAIN_CAP: the measured VRAM line (the served `kv_vram_cells`,
            which engine patch 0041 reports in /props, or YAMADORI_MAIN_CAP
            as the deploy wrote it). EVERY conversation is advertised this
            window (catalog.context_window) -- "I don't want to limit context
            to support it".
  helper    CHILD_TOKENS: the child's window (the second brain, the decider,
            side calls, an as-sent compaction; one slot, mcp/slots.py). It is
            NOT taken from main: it swaps into VRAM while its conversation's
            main pauses (engine 0041 ranks), so it can be 64k+ ("the child
            can be 64k+"). YAMADORI_CHILD_TOKENS, 65,536 by default (the
            operator's 64k; PHASE B measures the swap at 49k/64k/larger).
  window    the most any ONE request may occupy: the cap (a compaction's
            window included -- tiers.compaction_budget).
  the rest  of the pool (-c = 2 x cap + child, the deploy's arithmetic) is
            a second concurrent conversation's room, in host RAM when it
            must be: it spills, the primary does not (slots RANKS).

LAYOUT V2 (operator, 2026-09-29): "The point is to get more context at speed
in vram, so decider was the only thing that needed room." The child slot
becomes THE LANE (mcp/slots.py): the decider and small side calls only, kept between turns and
kept in VRAM at a rank above the primary conversation's (slots.RANK_LANE), so
its cells come OUT OF THE LINE:

  main      the line less the lane: MAIN_CAP (the deploy writes N - LANE),
            else the served `kv_vram_cells` less LANE_TOKENS while the lane
            is ranked (slots.lane_ranked()); advertised to every conversation
  lane      LANE_TOKENS (YAMADORI_LANE_TOKENS), below
  helper    CHILD_TOKENS still: the window a second-brain job gets while that
            machinery exists (it runs on the lane's slot at its
            conversation's rank; -c no longer reserves its 64k, so beside two
            conversations at the cap it has no room)
  child     the child slot's role for the dashboard (child(): "decider
            lane", its size, what it serves, kept / ranked)
  the pool  -c = 2N + LANE (the operator's formula; bench/deploy_layout_v2.py)

THE LANE'S SIZE, DERIVED (logs/proxy.log, read 2026-09-29): the slot holds one
request's prompt and generation at a time (requests queue there), so it must
hold the largest one. The decider: the last 200 decider turns held median 837,
p90 1,609, max 3,047 cells (each release's `cells_before`); its state is capped
at decide_turn.STATE_TOKENS = 2,048 by construction and the rest -- the system
block, the question and its options, the answer lead, one generated token --
is not capped: 3,047 - 2,048 = 999 cells at most in that sample. Side calls:
69 releases, median 439, p90 614, max 1,682 cells. So LANE_TOKENS =
STATE_TOKENS + 1,024 (the 999 rounded up to whole 256-cell blocks, the engine's
placement unit, 0038) = 3,072, which holds every request of both samples. What
does not fit is not refused (a unified pool gives every slot the whole window):
its extra cells land where the engine puts them, past the line if need be.

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

# Fractions of the discovered pool: 0.70 + 1 x 0.30 = the whole pool
# (operator, 2026-09-25, with the return to q8_0 KV at -c 163840: main
# 114,688, helper 49,152 -- "push second brain down to the 48k range").
# A CHOICE, not a measurement. Hermes compacts at 75% of the advertised main
# share (its small-window floor), so ~86K: the operator's "compact before or
# around 90k" (effective range ~90-120k, unmeasured). Deep thinking's hops
# are capped by job (tiers.JOB_THINKING). It replaced 5/8 + 3/8 (2026-09-22).
MAIN_SHARE = float(os.environ.get("YAMADORI_MAIN_SHARE", "0.70"))
HELPER_SHARE = float(os.environ.get("YAMADORI_HELPER_SHARE", "0.30"))
# THE HELPER AS A FIXED SIZE (operator, 2026-09-25: "~48k second brain is the
# constraint ... make sure the main model gets the share of the new tokens").
# When > 0 the helper gets exactly this many tokens and main gets the rest of
# the pool, so a larger -c goes to the conversation; the fractions above are
# then unused (0 restores them). At -c 181,248: main 132,096 + helper 49,152.
HELPER_TOKENS = int(os.environ.get("YAMADORI_HELPER_TOKENS", "49152"))
HELPERS = int(os.environ.get("YAMADORI_HELPERS", "1"))
# THE CAP LAYOUT (docstring). MAIN_CAP > 0 overrides the served line; 0 reads
# it from /props (`kv_vram_cells`, engine patch 0041); with neither the split
# above applies (a pool that is all VRAM).
MAIN_CAP = int(os.environ.get("YAMADORI_MAIN_CAP", "0") or 0)
CHILD_TOKENS = int(os.environ.get("YAMADORI_CHILD_TOKENS", "65536") or 0)
# LAYOUT V2's lane (docstring, "THE LANE'S SIZE, DERIVED"): STATE_TOKENS 2,048
# + 1,024 = 3,072 cells.
LANE_TOKENS = int(os.environ.get("YAMADORI_LANE_TOKENS", "3072") or 0)
# The conversation never gets less than this share however the fractions are
# set, because a helper starving the thing it was spawned to serve is the one
# outcome that makes the whole arrangement worse than not having it.
MAIN_FLOOR = float(os.environ.get("YAMADORI_MAIN_FLOOR", "0.50"))
# Measured on this hardware at --cache-type-k q8_0 --cache-type-v q8_0.
KV_KIB_PER_TOKEN = float(os.environ.get("YAMADORI_KV_KIB", "44"))

_POOL: int | None = None
# The tiered cache's VRAM line from the same /props answer (`kv_vram_cells`,
# 0 when the cache is not tiered, None when the server does not report it).
_LINE: int | None = None
# llama-server's slot count, from the same /props answer (`total_slots`), for
# mcp/slots.py. None until pool_size() has asked, or when it could not.
_SLOTS: int | None = None
# pool_size()'s answer while the pool has never been read (its docstring). RETRY_S: how often the proxy's startup
# thread asks again until a real read succeeds -- the proxy's own HEARTBEAT interval (5 s), so a model server that
# comes up is noticed within one beat.
FALLBACK_POOL = 131072
RETRY_S = 5.0
_READ = False


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
    global _POOL, _SLOTS, _LINE, _READ
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
                _READ = True
                if isinstance(d.get("total_slots"), int) and d["total_slots"] > 0:
                    _SLOTS = d["total_slots"]
                if isinstance(d.get("kv_vram_cells"), int):
                    _LINE = d["kv_vram_cells"]
                return _POOL
        except Exception:                                        # noqa: BLE001
            continue
    # A FAILED READ KEEPS THE LAST GOOD POOL. The dashboard's vitals refresh
    # (refresh=True) ran while the Flash-Next gate held the card
    # (2026-09-28): the model server did not answer, the fallback replaced
    # the real 262,144 pool, and every request of the proxy would have been
    # budgeted against 131,072 until the next good read. The fallback is
    # only for a process that has never read the pool. THE FALLBACK IS NOT THE
    # LAST WORD (2026-10-02): a proxy that started before the model server
    # answered kept 131,072 and no slot count for its whole life (deploy check
    # 2026-10-01: /v1/models said 131072 and requests were ranked for 3 slots
    # while bonsai served 209,920 cells on 1 slot, so ONE CONVERSATION was
    # off). read_ok() says whether a real read has happened; the proxy's
    # startup thread (server.py _read_pool_when_ready) keeps asking the main
    # model's own port every RETRY_S until one has -- off the request path, so
    # no request ever waits on a dead port.
    if _POOL is None:
        _POOL = FALLBACK_POOL
    return _POOL


def read_ok() -> bool:
    """Has this process READ the pool from a model server (not the fallback)?"""
    return _READ


def known_pool() -> int | None:
    """The pool this process has READ (pool_size's cache), or None when it
    has never read one. Asks nothing: the dashboard's readers use it so a
    view never makes the first read (docs/DASHBOARD.md, "A view never loads
    a model")."""
    return _POOL


def refresh_direct(timeout: float = 3) -> tuple[int | None, str]:
    """Re-read the pool from the main model's OWN server (DIRECT, config.yaml
    startPort) and nothing else -- never llama-swap, whose /upstream/<model>
    route would start a model that is off the card. Updates the same cache
    pool_size() keeps. (pool, how). A failed read keeps the last good pool
    (pool_size's rule) and says why."""
    global _POOL, _SLOTS, _LINE, _READ
    try:
        with urllib.request.urlopen(f"{DIRECT}/props", timeout=timeout) as r:
            d = json.load(r)
    except Exception as e:                                       # noqa: BLE001
        return _POOL, f"{DIRECT}/props: {type(e).__name__}"[:160]
    n = ((d.get("default_generation_settings") or {}).get("n_ctx")
         or d.get("n_ctx"))
    if not n:
        return _POOL, f"{DIRECT}/props carried no n_ctx"
    _POOL = int(n)
    _READ = True
    if isinstance(d.get("total_slots"), int) and d["total_slots"] > 0:
        _SLOTS = d["total_slots"]
    if isinstance(d.get("kv_vram_cells"), int):
        _LINE = d["kv_vram_cells"]
    return _POOL, f"{DIRECT}/props"


def lane_reserved() -> int:
    """The lane's cells taken out of the VRAM line: LANE_TOKENS while the
    lane is ranked above the primary (slots.lane_ranked(): YAMADORI_KV_RANK
    on), else 0 (no ranks are sent: the child then swaps like any slot)."""
    try:
        import slots
        return max(LANE_TOKENS, 0) if slots.lane_ranked() else 0
    except Exception:                                            # noqa: BLE001
        return 0


def main_cap(pool: int | None = None) -> tuple[int | None, str]:
    """(the main cap, where it came from): YAMADORI_MAIN_CAP (the deploy
    writes the line less the lane), else the served VRAM line (/props
    kv_vram_cells) less the lane (lane_reserved) when the pool is tiered
    past it, else (None, why) -- the split applies."""
    p = pool or pool_size()
    if MAIN_CAP > 0:
        return min(MAIN_CAP, p), "YAMADORI_MAIN_CAP"
    if _LINE and 0 < _LINE < p:
        lane = lane_reserved()
        if lane and _LINE > lane:
            return int(_LINE) - lane, (f"llama-server /props kv_vram_cells "
                                       f"less the lane ({lane})")
        return int(_LINE), "llama-server /props kv_vram_cells"
    return None, ("no cap: the pool is not tiered" if _LINE == 0 else
                  "no cap: the server reports no kv_vram_cells")


def model_window(model: str | None = None) -> dict | None:
    """ONE MODEL PER EFFORT TIER (mcp/tier_models.py): the window the table declares for the model this request is
    served by -- {ctx, slots, main_cap?, source} -- or None for the default model (its pool is read from its own
    /props, below) and with the table off. The deploy (bench/deploy_tier_models.py) writes each row from the same
    -c it writes into config.yaml, so the two agree by construction; nothing here reads another model's /props (it
    would load that model)."""
    try:
        import max_mode
        import tier_models
        t = tier_models.table()
        if not t.profiles_on:
            return None
        m = model or max_mode.current()
        if m == t.default:
            return None
        return t.window(m)
    except Exception:                                            # noqa: BLE001
        return None


def budgets(pool: int | None = None, model: str | None = None) -> dict:
    """The shares of the pool the request's MODEL serves (the per-model window: model_window), or the default
    model's (its served /props, the cap layout or the split)."""
    w = None if pool else model_window(model)
    if w:
        p = int(w["ctx"])
        cap = int(w["main_cap"]) if w.get("main_cap") else None
        source = f"tier table: {w.get('source') or 'window'}"
    else:
        p = pool or pool_size()
        cap, source = main_cap(p)
    if cap:
        # THE CAP LAYOUT (docstring): main = the cap, the child its own size
        # (it swaps in while its main pauses, so it is not taken from main).
        helper = max(min(CHILD_TOKENS or cap, p), 0)
        return {"pool": p, "main": cap, "helper": helper, "helpers": HELPERS,
                "lane": max(LANE_TOKENS, 0), "child": child(),
                "reserve": max(p - cap - helper, 0), "window": cap,
                "layout": "cap", "cap_source": source,
                "gib": round(p * KV_KIB_PER_TOKEN / 1024 / 1024, 2),
                **({"model_window": w} if w else {})}
    if HELPER_TOKENS > 0:
        helper = min(HELPER_TOKENS,
                     (p - int(p * MAIN_FLOOR)) // max(HELPERS, 1))
        main = p - max(HELPERS, 1) * helper
    else:
        main = max(int(p * MAIN_SHARE), int(p * MAIN_FLOOR))
        helper = int(p * HELPER_SHARE)
    if main + HELPERS * helper > p:
        helper = max((p - main) // max(HELPERS, 1), 0)
    return {"pool": p, "main": main, "helper": helper, "helpers": HELPERS,
            "child": child(),
            "reserve": p - main - HELPERS * helper, "window": p,
            "layout": "split", "cap_source": source,
            "gib": round(p * KV_KIB_PER_TOKEN / 1024 / 1024, 2),
            **({"model_window": w} if w else {})}


def child() -> dict:
    """The child slot's role and size, named (layout v2): the dashboard's KV
    panel prints it instead of a label of its own, which still said "deep
    thinking" (docs/DASHBOARD.md). {role, tokens, serves, kept, ranked, rank,
    slot}."""
    try:
        import slots
        kept, ranked, rank = slots.lane_kept(), slots.lane_ranked(), slots.RANK_LANE
        slot = slots.child_slot(slots._known_n())
    except Exception:                                            # noqa: BLE001
        kept = ranked = False
        rank = slot = None
    return {"role": "decider lane", "tokens": max(LANE_TOKENS, 0),
            "serves": ["the decider", "small side calls (titles)"],
            "also": "a second-brain job, when one runs, at its conversation's "
                    "rank with the helper window",
            "not": "a compaction sent up as is (the least recently used "
                   "conversation slot)",
            "kept": kept, "ranked": ranked,
            "rank": rank if ranked else None, "slot": slot}


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
    how = (f"fixed {HELPER_TOKENS} each" if HELPER_TOKENS > 0
           else f"{HELPER_SHARE:.0%} each")
    print(f"    main   {b['main']:>7}  (the rest; floor {MAIN_FLOOR:.0%})"
          if HELPER_TOKENS > 0 else
          f"    main   {b['main']:>7}  ({MAIN_SHARE:.0%}, floor {MAIN_FLOOR:.0%})")
    print(f"    helper {b['helper']:>7}  ({how}, x{b['helpers']})")
    print(f"    reserve{b['reserve']:>7}  (whatever the shares leave unclaimed)")
    print()
    print(what_if())
    print("\n  Unified KV has no per-slot reservation, so these are enforced")
    print("  HERE, by capping what each kind of request may send. The server")
    print("  would happily let a helper consume the conversation's tokens.")
