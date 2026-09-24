#!/usr/bin/env python
"""Which llama-server slot a request lands on.

THE FAILURE THIS EXISTS FOR

llama-server runs four slots (config.yaml sets no `-np`, so the build's auto
default applies: n_parallel 4 over one unified KV pool; server.cpp logs
"n_parallel is set to auto, using n_parallel = 4 and kv_unified = true"). A
slot keeps the tokens of the last prompt it served, and a new request whose
prompt shares a prefix with them skips that prefix: live, a Hermes turn reused
37,791 of 40,080 prompt tokens. Left to itself the server picks a slot by
prompt similarity, else the least recently used idle one
(server-context.cpp, get_available_slot). So:

  - a conversation's next turn can land on another slot and re-prefill
    20-40k tokens from nothing, and
  - a side call -- a one-word approval check, a compaction -- takes whichever
    idle slot was used longest ago, which is often the one holding the
    conversation's cached prefix, and evicts it.

THE RULE

  pinned      a conversation keeps one slot for as long as it is used: its
              turns go back to the slot that already holds their prefix.
  helper      the second brain (key HELPER: deep thinking, fan-out's B/C,
              fix-ups, summaries) always uses its RESERVED slot,
              helper_slot() = n-2, and never a conversation's.
  transient   a request that is not a conversation -- a client utility call
              -- goes to a slot NO conversation holds. The highest slot is
              never pinned, so there is always one. It never takes a pin.
              (A compaction runs on its conversation's slot, as part of that
              conversation -- mcp/compaction.py; one it cannot place that way
              falls back to COMPACTION AFFINITY below.)
  busy        a slot this process has a request in flight on is never chosen:
              llama-server DEFERS a request pinned to a busy slot (it waits
              for that slot, server-context.cpp "requested slot is
              unavailable, defer task"), which would turn a retried turn into
              a wait for the orphan it replaces. A conversation whose own
              slot is busy moves to a free one.
  LRU         more live conversations than pinnable slots: the one used
              longest ago loses its pin (its cache is what gets overwritten).

The request carries `id_slot` (llama-server's field, server-context.cpp
`task.id_slot = json_value(data, "id_slot", -1)`, copied through the OpenAI
chat parse as an unknown field) and `cache_prompt: true` (the server's default,
server-schema.cpp; sent so a changed default cannot silently turn reuse off).

WHAT THIS DOES NOT SEE

Other processes. The worker (mcp/worker.py) and the tools API call the model
through mcp/model.py in their own processes with their own copies of this
table; their calls go to HELPER there. So the second brain's slot is a FIXED
number every process computes the same way -- helper_slot(): n-2, slot 2 of
4 -- and RESERVED: no conversation is ever pinned to it (2026-09-24, #10/#11
in docs/SELF-IMPROVEMENT-LOG.md). Before that, HELPER was an ordinary LRU pin
that took slot 2 "when free", conversations filled slot 2 when 0 and 1 were
held, and two things followed: another process's second brain landed on a
conversation's slot 2 and overwrote it, and in this process the second brain
(which runs deep thinking BEFORE the conversation's own turn acquires its
slot) could evict the pin of the very conversation it was working for,
because that conversation's last use was its previous turn. The cost of the
reservation: with 4 slots, two conversations hold a pin at once, not three.
The server defers a second-brain call while that slot is busy, so the cost of
two second-brain calls at once is a wait, never a conversation's cache. Set
YAMADORI_SLOT_PINNING=0 to hand every choice back to llama-server.
"""
from __future__ import annotations

import json
import os
import threading
import time

ENABLED = os.environ.get("YAMADORI_SLOT_PINNING", "1") == "1"
# llama-server's auto default when -np is not set (server.cpp:152-155).
FALLBACK_SLOTS = 4
HELPER = "helper"

_lock = threading.Lock()
_n: int | None = (int(os.environ["YAMADORI_SLOTS"])
                  if os.environ.get("YAMADORI_SLOTS", "").isdigit() else None)
_n_source = "YAMADORI_SLOTS" if _n else ""
_pins: dict[str, int] = {}        # key -> slot
_used: dict[str, float] = {}      # key -> last time it was given its slot
_busy: dict[int, int] = {}        # slot -> requests in flight from THIS process
# slot -> how many of those are WARMS (proxy._warm): a zero-token request that
# loads a conversation's next prefix while its harness runs a tool. A slot
# busy only with its own conversation's warm stays that conversation's: its
# next turn is sent there anyway and llama-server defers it until the warm
# ends (server-context.cpp "requested slot is unavailable, defer task"),
# which costs the rest of the warm and keeps the prefix it just loaded.
# Moving it to another slot would cost the whole prompt.
_warming: dict[int, int] = {}
# slot -> fingerprint() of the last prompt this process completed there. See
# COMPACTION AFFINITY below. Dropped the moment the slot is granted again (its
# content is about to change) and written back only when that request ends
# with an answer, so a failed or foreign request leaves no stale claim.
_prompts: dict[int, dict] = {}


def count(refresh: bool = False) -> int:
    """How many slots the server runs: `total_slots` from the /props answer
    mcp/budget.py already fetches for the pool size (no second probe).

    Falls back to FALLBACK_SLOTS -- the build's auto default -- and says so in
    `source()`, because a wrong count only misplaces pins (an id_slot past the
    end wraps, server-context.cpp get_slot_by_id). A fallback is not cached,
    so the real count replaces it once /props has answered."""
    global _n, _n_source
    if _n is not None and not refresh:
        return _n
    try:
        import budget
        budget.pool_size()
        n = budget._SLOTS
    except Exception:                                            # noqa: BLE001
        n = None
    if n:
        _n, _n_source = int(n), "llama-server /props total_slots"
        return _n
    _n_source = "fallback: /props gave no total_slots"
    return FALLBACK_SLOTS


def source() -> str:
    return _n_source


def reset(n: int | None = None) -> None:
    """Forget every pin (tests; a changed server). `n` sets the slot count."""
    global _n, _n_source
    with _lock:
        _pins.clear()
        _used.clear()
        _busy.clear()
        _warming.clear()
        _prompts.clear()
        if n is not None:
            _n, _n_source = int(n), "reset"


def _transient_slot(n: int) -> int | None:
    """The slot kept free of pins, or None when there is only one slot."""
    return n - 1 if n >= 2 else None


def helper_slot(n: int | None = None) -> int | None:
    """The second brain's reserved slot: n-2, the same number in every
    process (see WHAT THIS DOES NOT SEE). None below three slots, where
    reserving one would leave a conversation nowhere to live; the second
    brain then shares the transient slot."""
    n = count() if n is None else n
    return n - 2 if n >= 3 else None


# ---------------------------------------------------------------------------
# COMPACTION AFFINITY -- the FALLBACK for a compaction.
#
# A compaction is served on its conversation's stored prompt and slot by
# mcp/compaction.py (proxy._serve_compaction): in place, or rewritten from
# Hermes' flattened form. When that finds nothing -- a restart, another
# account, a transcript that does not map -- the request goes up as sent,
# and this places it: on the pinned slot whose last prompt shares the longest
# prefix with it, when that prefix is worth taking the slot for, and
# otherwise on the transient slot. It USES the slot and nothing else: the
# pin and its LRU time are untouched.
#
# WHAT IS COMPARED: the prompt as this process SENT it upstream (tools and
# system message as segment 0, then one segment per message), as a chain of
# cumulative hashes -- fingerprint(). Chained, so segment i matches only when
# every segment before it matched too, which is what a KV prefix is. It is
# what the slot holds up to the template's rendering and the generated reply.
#
# WHY NOT llama-server's GET /slots. Its per-slot `prompt` (the detokenized
# cached prompt) is written only when LLAMA_SERVER_SLOTS_DEBUG is set
# (server-context.cpp server_slot::to_json, `if (!only_metrics)`), which
# config.yaml does not set; without it /slots carries counts, not content. And
# a token-level compare would cost a /tokenize of the compaction's prompt --
# tens of thousands of tokens -- per call. The hash chain costs a json.dumps
# and a sha1 per message, in process. What it cannot see: another process's
# requests on a slot (the worker pins to HELPER, which is never a candidate),
# and llama-server purging an idle slot under KV pressure
# (server-context.cpp try_clear_idle_slots). Either shows up as a low
# `reused` in x_yamadori.cache, which is the measurement; this is the guess.
#
# A FLATTENED COMPACTION SENT AS IS SHARES NOTHING. Hermes' (corpus, 12 of
# 12) is ONE user message -- "You are a summarization agent creating a
# context checkpoint ... TURNS TO SUMMARIZE: [USER]: ..." -- with no system
# prompt and no tools, while its conversation's prompt opens with ~40 tools,
# which the served chat template renders FIRST in the system block (the
# GGUF's tokenizer.chat_template: `<|im_start|>system\n` + effort line +
# `# Tools` + tool JSON + system message). So as sent it matches no slot and
# goes to the transient one -- sending it to the conversation's slot would
# reuse nothing and overwrite the tools-and-system prefix the next turn
# reuses. That is why compaction.py rewrites it onto the stored prompt
# instead, and this is only what happens when it cannot.
AFFINITY_MIN_TOKENS = int(os.environ.get("YAMADORI_COMPACTION_AFFINITY_MIN",
                                         "1024"))


def _segment(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def fingerprint(payload: dict) -> dict:
    """{hashes, chars}: the cumulative hash and character count of the prompt
    after each segment -- tools with the system message, then each message."""
    import hashlib
    msgs = [m for m in (payload.get("messages") or []) if isinstance(m, dict)]
    head = None
    if msgs and msgs[0].get("role") in ("system", "developer"):
        head, msgs = msgs[0], msgs[1:]
    keep = ("role", "content", "reasoning_content", "tool_calls",
            "tool_call_id", "name")
    # The served template's system block renders, in order: the effort line
    # (thinking on and effort xhigh or low only), the tool list, then the
    # system message. So a request with other tools, or tools where the slot
    # had none, shares nothing past `<|im_start|>system\n`.
    thinking = payload.get("enable_thinking") is not False and (
        (payload.get("chat_template_kwargs") or {}).get("enable_thinking")
        is not False)
    effort = (payload.get("reasoning_effort") or "xhigh") if thinking else None
    segs = [_segment({"effort_line": effort if effort in ("xhigh", "low")
                      else None,
                      "tools": payload.get("tools") or [],
                      "system": ({k: head.get(k) for k in keep if k in head}
                                 if head else None)})]
    segs += [_segment({k: m.get(k) for k in keep if k in m}) for m in msgs]
    hashes, chars = [], []
    h, c = "", 0
    for s in segs:
        h = hashlib.sha1((h + s).encode("utf-8", "replace")).hexdigest()
        c += len(s)
        hashes.append(h)
        chars.append(c)
    return {"hashes": hashes, "chars": chars}


def shared_prefix(a: dict | None, b: dict | None) -> int:
    """Characters of prompt two fingerprints share from the start (whole
    segments only; 0 when segment 0 differs)."""
    if not a or not b:
        return 0
    n = 0
    for x, y in zip(a.get("hashes") or [], b.get("hashes") or []):
        if x != y:
            break
        n += 1
    return int(a["chars"][n - 1]) if n else 0


def remember(grant: dict | None, fp: dict | None) -> None:
    """A request finished on this slot with an answer: it now holds `fp`."""
    if grant and grant.get("slot") is not None and fp:
        with _lock:
            _prompts[grant["slot"]] = fp


def _affinity(prefix: dict, busy: set) -> dict:
    """The pinned conversation slot sharing the longest prefix with `prefix`.
    Called under _lock."""
    best = {"key": None, "slot": None, "shared_tokens": 0, "candidates": 0}
    for k, s in _pins.items():
        if k == HELPER or s in busy or s not in _prompts:
            continue
        best["candidates"] += 1
        # Tokens estimated as tiers.estimate_prompt_tokens does: chars / 3.
        tok = shared_prefix(prefix, _prompts[s]) // 3
        if tok > best["shared_tokens"]:
            best.update(key=k, slot=s, shared_tokens=tok)
    best["took"] = bool(best["slot"] is not None
                        and best["shared_tokens"] >= AFFINITY_MIN_TOKENS)
    best["min_tokens"] = AFFINITY_MIN_TOKENS
    return best


def acquire(key: str | None, transient: bool = False,
            prefix: dict | None = None, warm: bool = False) -> dict:
    """Choose a slot and count a request in flight on it.

    Returns {slot, mode, how, evicted}: `slot` is None when every slot is busy
    (the server then picks, as it did before pinning). Always pair with
    release(grant).

    `prefix` (a fingerprint(), transient requests only): try COMPACTION
    AFFINITY first. The grant then carries `affinity` -- {key (8 chars),
    slot, shared_tokens, candidates, took, min_tokens} -- whether or not a
    slot was taken, so the record says why it went where it went."""
    if not ENABLED:
        return {"slot": None, "mode": "off", "how": "pinning disabled",
                "evicted": None}
    if warm:
        # A warm goes to the conversation's own pinned slot or nowhere: it
        # exists to load THAT slot's next prefix.
        with _lock:
            slot = _pins.get(key) if key else None
            if slot is None:
                return {"slot": None, "mode": "warm", "how": "no pinned slot "
                        "to warm", "evicted": None, "warm": True}
            _busy[slot] = _busy.get(slot, 0) + 1
            _warming[slot] = _warming.get(slot, 0) + 1
            _prompts.pop(slot, None)
        return {"slot": slot, "mode": "warm", "how": "warm: this "
                "conversation's slot", "evicted": None, "warm": True}
    n = count()
    keep = _transient_slot(n)
    hs = helper_slot(n)
    now = time.time()
    evicted = None
    aff = None
    if key == HELPER and not transient:
        # THE SECOND BRAIN: its reserved slot, busy or not (the server defers
        # it behind the other second-brain call: a wait, never a
        # conversation's cache), and it never takes or evicts a pin. Below
        # three slots it shares the transient slot.
        slot = hs if hs is not None else keep
        with _lock:
            if slot is not None:
                _busy[slot] = _busy.get(slot, 0) + 1
                _prompts.pop(slot, None)
        return {"slot": slot, "mode": "helper",
                "how": ("helper: the second brain's reserved slot"
                        if hs is not None else
                        "helper: fewer than 3 slots; the transient slot"),
                "evicted": None}
    with _lock:
        busy = {s for s, c in _busy.items() if c > 0}
        held = {s: k for k, s in _pins.items()}
        if hs is not None:
            # Never a conversation's, never a transient call's first choice.
            held.setdefault(hs, HELPER)

        def lru_victim(exclude: str | None = None):
            cands = [(_used.get(k, 0.0), k) for k, s in _pins.items()
                     if s not in busy and k != exclude]
            return min(cands)[1] if cands else None

        if (transient or not key) and prefix:
            aff = _affinity(prefix, busy)
        if aff and aff["took"]:
            slot, mode = aff["slot"], "affinity"
            how = (f"affinity: shares ~{aff['shared_tokens']} prompt tokens "
                   f"with conversation {aff['key'][:8]}'s slot; its pin is "
                   f"unchanged")
        elif transient or not key:
            # The transient slot, else the helper's while it is idle (its
            # cache is one job's, cheap to lose), before any conversation's.
            free = [s for s in range(n - 1, -1, -1)
                    if s not in busy and s not in held]
            if not free and hs is not None and hs not in busy:
                free = [hs]
            if free:
                slot, how = free[0], "transient: a slot no conversation holds"
            else:
                victim = lru_victim()
                if victim is None:
                    slot, how = None, "every slot is busy; the server chooses"
                else:
                    slot = _pins.pop(victim)
                    _used.pop(victim, None)
                    evicted = victim
                    how = "transient: every slot pinned, took the least recently used"
            mode = "transient"
        else:
            mine = _pins.get(key)
            only_warm = (mine is not None and mine in busy
                         and _warming.get(mine, 0) >= _busy.get(mine, 0))
            if mine is not None and (mine not in busy or only_warm):
                slot, how = mine, ("pinned: this conversation's slot"
                                   + (" (after its warm)" if only_warm else ""))
            else:
                # Conversations fill from the low end, never onto the
                # transient slot or the second brain's reserved one (`held`
                # carries it).
                free = [s for s in range(n) if s != keep
                        and s not in busy and s not in held]
                if free:
                    slot = free[0]
                else:
                    victim = lru_victim(exclude=key)
                    slot = _pins.pop(victim) if victim is not None else None
                    if victim is not None:
                        _used.pop(victim, None)
                        evicted = victim
                if slot is None:
                    # The pin, if any, stays: that slot still holds the prefix.
                    how = "every pinnable slot is busy; the server chooses"
                else:
                    how = ("moved: this conversation's slot is busy with an "
                           "earlier request" if mine is not None else
                           "pinned: first turn on this slot")
                    if evicted:
                        how += " (evicted the least recently used conversation)"
                    _pins[key] = slot
            mode = "pinned"
            _used[key] = now
        if slot is not None:
            _busy[slot] = _busy.get(slot, 0) + 1
            # Its content is about to change; remember() writes it back.
            _prompts.pop(slot, None)
    grant = {"slot": slot, "mode": mode, "how": how,
             "evicted": (evicted or "")[:8] or None}
    if aff is not None:
        grant["affinity"] = dict(aff, key=(aff["key"] or "")[:8] or None)
    return grant


def release(grant: dict | None) -> None:
    if not grant or grant.get("slot") is None:
        return
    with _lock:
        s = grant["slot"]
        _busy[s] = max(_busy.get(s, 0) - 1, 0)
        if grant.get("warm"):
            _warming[s] = max(_warming.get(s, 0) - 1, 0)


def snapshot() -> dict:
    with _lock:
        return {"slots": _n, "source": _n_source,
                "pins": {k[:8]: s for k, s in _pins.items()},
                "helper": helper_slot(_n) if _n else None,
                "busy": {s: c for s, c in _busy.items() if c},
                "known_prompts": {s: fp["chars"][-1] if fp.get("chars") else 0
                                  for s, fp in _prompts.items()}}


def cache_record(timings: dict | None, usage: dict | None,
                 grant: dict | None) -> dict:
    """{prompt, reused, processed, slot, mode} for one upstream generation.

    From llama-server's `timings` (server-common.cpp server_slot_stats:
    `cache_n` = prompt tokens taken from the slot's cache, `prompt_n` = prompt
    tokens processed now); else from usage.prompt_tokens_details.cached_tokens;
    else unknown (None), never a guessed zero."""
    t = timings or {}
    u = usage or {}
    reused = t.get("cache_n")
    processed = t.get("prompt_n")
    if reused is None and processed is None:
        details = u.get("prompt_tokens_details") or {}
        if "cached_tokens" in details and u.get("prompt_tokens") is not None:
            reused = int(details.get("cached_tokens") or 0)
            processed = int(u["prompt_tokens"]) - reused
    prompt = (int(reused) + int(processed)
              if reused is not None and processed is not None
              else u.get("prompt_tokens"))
    g = grant or {}
    rec = {"prompt": prompt, "reused": reused, "processed": processed,
           "slot": g.get("slot"), "mode": g.get("mode"),
           "evicted": g.get("evicted")}
    if g.get("affinity"):
        rec["affinity"] = g["affinity"]
    return rec


if __name__ == "__main__":
    print(json.dumps({"slots": count(), "source": source()}, indent=1))
