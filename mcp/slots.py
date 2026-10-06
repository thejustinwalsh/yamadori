#!/usr/bin/env python
"""Which llama-server slot a request lands on.

THE FAILURE THIS EXISTS FOR

llama-server runs its slots over one unified KV pool (THREE since 2026-09-28,
config.yaml `-np 3`: two conversation slots and one child slot, operator; four
before, the build's auto default). A
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
              Conversations live on slots 0 .. n-2 (conversation_slots()).
  child       THE CHILD SLOT, child_slot() = n-1, is everything that is not
              a conversation's own turn (operator, 2026-09-28: "3 slots = two
              conversation slots + one child slot"): the second brain (key
              HELPER: deep thinking, the plan, fan-out's B/C, fix-ups,
              summaries), the skill decider, a client's side calls and a
              compaction sent up as is. helper_slot() and the transient slot
              ARE the child slot. They QUEUE on it: a request for it goes
              there busy or not, and llama-server runs them one at a time
              (it defers a task pinned to a busy slot, server-context.cpp
              "requested slot is unavailable, defer task"). It never takes a
              pin and never evicts one.
  transient   a request that is not a conversation -- a client utility call
              -- goes to the child slot (above).
              (A compaction runs on its conversation's slot, as part of that
              conversation -- mcp/compaction.py; one it cannot place that way
              falls back to COMPACTION AFFINITY below, and then, in layout
              v2, to the least recently used CONVERSATION slot.)
  the lane    LAYOUT V2 (operator, 2026-09-29: "The point is to get more
              context at speed in vram, so decider was the only thing that
              needed room"): the child slot is THE LANE -- the decider and
              small side calls (titles) only (a second-brain job, while that
              machinery exists, still runs there at its conversation's
              rank). It is budget.LANE_TOKENS cells, NOT released after a
              decider turn or a side call (its head stays cached: the
              releases cost median 395 ms, p90 824 ms each, logs/proxy.log,
              200 decider turns), and ranked above the primary conversation
              (RANK_LANE) so the engine never moves it out of VRAM; its cells
              come out of the main cap (budget.main_cap). A compaction sent
              up as is (thousands of tokens; 150,659 in the proxy log) goes
              to the least recently used conversation slot instead
              (coordinator, 2026-09-29: a cache miss beats a failed
              compaction) and that slot is emptied after it. THE LANE below.
  busy        a CONVERSATION slot this process has a request in flight on is
              never chosen for another conversation's turn:
              llama-server DEFERS a request pinned to a busy slot (it waits
              for that slot, server-context.cpp "requested slot is
              unavailable, defer task"), which would turn a retried turn into
              a wait for the orphan it replaces. A conversation whose own
              slot is busy moves to a free one.
  LRU         more live conversations than pinnable slots: the one used
              longest ago loses its pin (its cache is what gets overwritten).
  adopted     a conversation key with no pin (its key changed) takes the idle
              slot whose prompt its own continues, before a free one
              (ADOPTION below); pins survive a proxy restart (`persist`).
  ranked      every request names each slot's KV RANK (RANKS below): the
              primary conversation's slot outranks the other conversation's,
              and the child takes the rank of the conversation it works for,
              so the engine (engines/patches/llama-bonsai2-ada/0041) keeps
              the primary's cells below the tiered cache's VRAM line and
              moves a second concurrent conversation out first.

The request carries `id_slot` (llama-server's field, server-context.cpp
`task.id_slot = json_value(data, "id_slot", -1)`, copied through the OpenAI
chat parse as an unknown field), `cache_prompt: true` (the server's default,
server-schema.cpp; sent so a changed default cannot silently turn reuse off),
and `kv_rank` / `kv_ranks` (0041's fields; upstream_fields()).

WHAT THIS DOES NOT SEE

Other processes. The worker (mcp/worker.py) and the tools API call the model
through mcp/model.py in their own processes with their own copies of this
table; their calls go to HELPER there. So the second brain's slot is a FIXED
number every process computes the same way -- helper_slot() = child_slot() =
n-1 (slot 2 of 3; n-2 of 4 until 2026-09-28) -- and RESERVED: no conversation
is ever pinned to it (2026-09-24, #10/#11
in docs/SELF-IMPROVEMENT-LOG.md). Before that, HELPER was an ordinary LRU pin
that took slot 2 "when free", conversations filled slot 2 when 0 and 1 were
held, and two things followed: another process's second brain landed on a
conversation's slot 2 and overwrote it, and in this process the second brain
(which runs deep thinking BEFORE the conversation's own turn acquires its
slot) could evict the pin of the very conversation it was working for,
because that conversation's last use was its previous turn. With 3 slots two
conversations hold a pin at once, and the side calls share the child slot
with the second brain instead of a slot of their own (they queue: a side call
waits for the second brain's hop in flight, never the other way round -- and a
side call that lands between two hops of a job costs the next hop the job's
shared head, which the server's host-RAM prompt cache may give back).
The server defers a second-brain call while that slot is busy, so the cost of
two second-brain calls at once is a wait, never a conversation's cache. Set
YAMADORI_SLOT_PINNING=0 to hand every choice back to llama-server.

A PIN IS ONLY AS GOOD AS THE SERVER KEEPING THE SLOT. With the build's
defaults (auto -np: 4 slots, unified KV, --cache-idle-slots ON) every task
that starts on ANY slot saves each idle slot to host RAM and clears it, and a
pinned request to the emptied slot loads nothing back: the helper's fix-up
or a side call on the transient slot wiped the conversation's slot (live gate
2026-09-24, docs/SELF-IMPROVEMENT-LOG.md #11). config.yaml sets
--no-cache-idle-slots on `bonsai` for that reason. `/slots` n_prompt_tokens
shows what each slot still holds.

AN IDLE SLOT'S CACHE IS NOT FREE (2026-09-26). With the unified KV pool every
decode step attends over the pool up to its highest used cell
(llama-kv-cache.cpp get_n_kv: GGML_PAD(cells.used_max_p1())), so a slot
nobody is using still slows every other slot's decode. Measured on the running
stack (bonsai, -c 181248, q8_0 KV, 4 slots; the kv_share_probe, n=2 per cell):
decode tok/s on slot 0 with the other three slots shrunk to ~0 vs holding
~40k tokens each -- 8k context 60.7/47.3 vs 20.1/18.2; 32k 42.8/43.2 vs
15.0/17.8; 64k 26.4/31.5 vs 14.3/15.9. So the slots whose cache nobody will
reuse are RELEASED after use (RELEASE, below): the second brain's after each
run, the transient slot after each side call. A conversation's pinned slot
is cleared only when it has been idle past IDLE_CLEAR_S and another request
is about to generate (IDLE CLEAR, below); its pin is kept.
"""
from __future__ import annotations

import json
import os
import threading
import time
import weakref

ENABLED = os.environ.get("YAMADORI_SLOT_PINNING", "1") == "1"
# The count when /props has not answered: config.yaml's `-np 3` (operator,
# 2026-09-28: two conversation slots + one child slot). It was 4, the build's
# auto default when no -np is set (server.cpp:152-155).
FALLBACK_SLOTS = 3
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
# RELEASE (below): slots whose cache this process filled as the second brain
# or a transient call since they were last released -- only those are worth a
# release -- and the releases in flight (acquire waits for one).
_dirty: set[int] = set()
_releasing: dict[int, threading.Event] = {}
_released_recent: list[dict] = []
# IDLE CLEAR (below): when each slot's last request or warm of this process
# ended, when each conversation last showed up (bind_request), the pinned
# slots cleared while idle (slot -> what the next request's record says),
# and compactions sent up as is in flight (attributed to no conversation).
_T0 = time.time()
_last_end: dict[int, float] = {}
_last_seen: dict[str, float] = {}
_cleared: dict[int, dict] = {}
_compacting = 0
_key_account: dict[str, str] = {}
# THE OTHER CARD (operator, 2026-09-30, verbatim: "If anything it would be ensuring that a second message that came
# in out of order got the other card, like a 60s timeout before you can assign a new conversation to it makes sense,
# if it doesn't match the current id"): a conversation whose id is not the main card's owner, arriving while the
# owner is mid-request or within PRIMARY_HOLD_S of its latest activity, is ROUTED to the other card -- the tier
# table's `other_card` for its model (bonsai-a4000) -- and keeps it for its life. One conversation there too: its
# slot 0 (OTHER_CONV_SLOT); jjava and side calls on that server use slot 1 (OTHER_LANE_SLOT). The same hold rule
# decides who may take it. `_other` is its one owner; `_displaced` the conversations a switch took a card from.
# What a SWITCH does to the previous owner's cells (THE OTHER CARD, below): llama-server saves a slot's prompt to
# its host-RAM prompt cache when a new prompt on that slot diverges from it (get_available_slot's save; the same
# mechanism IDLE CLEAR's one-token prompt relies on, measured 2026-09-27: ~41k tokens restored in 516 ms) and loads
# it back when that prompt returns. The save runs inside the newcomer's first prefill (its prompt_ms includes it);
# the restore is the returning conversation's resumed_cold {how, prompt_ms}.
SWITCH_SAVE = ("llama-server host-RAM prompt cache (--cache-ram): saved when this prompt diverges from it, restored "
               "if it returns (its resumed_cold says how and in how many ms)")
OTHER_CONV_SLOT = 0
OTHER_LANE_SLOT = 1
_other: dict = {"owner": None, "model": None, "busy": 0, "warming": 0, "last_end": 0.0}
_displaced: dict[str, dict] = {}


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
    global _n, _n_source, _compacting, _primary
    with _lock:
        _act.clear()
        _slot_for.clear()
        _helper_on.clear()
        _primary = None
        _pins.clear()
        _used.clear()
        _busy.clear()
        _warming.clear()
        _prompts.clear()
        _dirty.clear()
        _releasing.clear()
        _released_recent.clear()
        _last_end.clear()
        _last_seen.clear()
        _cleared.clear()
        _key_account.clear()
        _hold_for.clear()
        _answered.clear()
        _restore_stamp.clear()
        _other.update(owner=None, model=None, busy=0, warming=0, last_end=0.0)
        _displaced.clear()
        _compacting = 0
        if n is not None:
            _n, _n_source = int(n), "reset"


def child_slot(n: int | None = None) -> int | None:
    """THE CHILD SLOT: n-1, the same number in every process (see WHAT THIS
    DOES NOT SEE) -- the second brain, the decider, side calls and a
    compaction sent up as is. None with one slot (the server then chooses)."""
    n = count() if n is None else n
    return n - 1 if n >= 2 else None


def conversation_slots(n: int | None = None) -> list[int]:
    """The slots a conversation may be pinned to: all but the child's."""
    n = count() if n is None else n
    return list(range(n - 1)) if n >= 2 else list(range(n))


def _transient_slot(n: int) -> int | None:
    """A side call's slot: the child slot."""
    return child_slot(n)


def helper_slot(n: int | None = None) -> int | None:
    """The second brain's slot: the child slot (n-1 since 2026-09-28; n-2 of
    four before, with its own transient slot above it)."""
    return child_slot(n)


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
    # The segment of the first assistant turn (ADOPTION, below): a prompt
    # that shares a slot's prefix THROUGH an assistant turn -- text the model
    # generated for that conversation -- is that conversation.
    first_assistant = next((i + 1 for i, m in enumerate(msgs)
                            if m.get("role") == "assistant"), None)
    return {"hashes": hashes, "chars": chars,
            "first_assistant": first_assistant}


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


def _shared_segments(a: dict | None, b: dict | None) -> int:
    n = 0
    for x, y in zip((a or {}).get("hashes") or [], (b or {}).get("hashes") or []):
        if x != y:
            break
        n += 1
    return n


# AN UNANSWERED ATTEMPT HOLDS NOTHING (2026-10-02; the operator's live session of 2026-10-01 17:28-17:35 ET, VS
# Copilot at max): the first request's client hung up 180 s into the Flash-Next load, before any byte; its retries
# arrived as NEW conversation ids (a retried opening mints another: session_identity) and were refused 503
# conversation_at_capacity for the next 60 s -- held off by the cancelled attempt's own pin ("its latest activity
# was 4 s ago"). A conversation becomes the card's owner when a generation of it FINISHES (remember(), below); a
# pin made by an attempt that ended with none -- cancelled, refused, failed -- is dropped at release, with its
# activity stamps, so the card is free for the retry. A conversation answered before keeps its pin whatever
# happens to a later request.
_answered: set[str] = set()


def remember(grant: dict | None, fp: dict | None) -> None:
    """A request finished on this slot with an answer: it now holds `fp`."""
    if grant and grant.get("_key"):
        with _lock:
            grant["_answered"] = True
            _answered.add(grant["_key"])
    if grant and grant.get("slot") is not None and fp and not grant.get("_server"):
        with _lock:
            _prompts[grant["slot"]] = fp
        _save()


# ---------------------------------------------------------------------------
# ADOPTION: a conversation key with no pin that CONTINUES a slot's prompt.
#
# (docs/SELF-IMPROVEMENT-LOG.md #38, operator 2026-09-25: "why does it jump
# around?") A pin is keyed by the conversation's session key, and a key can
# change while the conversation goes on: a compaction continuation the link
# missed, the first two messages changing (nebari.key_of's KNOWN GAP: a
# conversation with no system message is keyed [user] on turn 1 and [user,
# assistant] on turn 2). The new key had no pin, so it took a free slot or
# evicted the least recently used conversation -- and the slot holding its
# own prefix was left pinned to the old key, abandoned, its cells still in
# the unified KV (see the note on used cells in #38).
#
# So a key with no pin first looks at what each idle pinned slot last held
# (_prompts, the same fingerprints COMPACTION AFFINITY reads) and ADOPTS the
# slot whose prompt this one CONTINUES: it shares that prompt through one of
# its assistant turns (text the model generated for that conversation, which
# no other conversation carries), or it strictly extends the whole of it. A
# shared harness head -- tools and system prompt, often >1,024 tokens -- is
# never enough: two conversations of one harness share it, and taking each
# other's slots on it would make them thrash. The old key's pin moves to the
# new key. At least AFFINITY_MIN_TOKENS shared (a choice, as there).
# ---------------------------------------------------------------------------
def _adopt(prompt: dict, busy: set) -> dict | None:
    """The idle pinned slot whose prompt `prompt` continues, or None. Called
    under _lock."""
    best = None
    for k, s in _pins.items():
        if k == HELPER or s in busy or s not in _prompts:
            continue
        held = _prompts[s]
        n = _shared_segments(prompt, held)
        if not n:
            continue
        fa = held.get("first_assistant")
        through_answer = fa is not None and n > fa
        extends = (n == len(held.get("hashes") or [])
                   and len(prompt.get("hashes") or []) > n)
        tok = int(held["chars"][n - 1]) // 3
        if (through_answer or extends) and tok >= AFFINITY_MIN_TOKENS and (
                best is None or tok > best["shared_tokens"]):
            best = {"key": k, "slot": s, "shared_tokens": tok,
                    "why": "through an answer" if through_answer
                    else "extends its whole prompt"}
    return best


# ---------------------------------------------------------------------------
# PINS SURVIVE A PROXY RESTART (#38). _pins and _prompts lived in this
# process only, so every restart -- every deploy -- sent every live
# conversation to whichever slot was free first, in arrival order, while
# llama-server still held each one's prefix on its old slot. `persist(path)`
# (called by server.py at startup, never on import, so no test writes it)
# loads the table from `path` and saves it after every change. A pin to a
# slot llama-server has since emptied (a llama-swap restart) costs what it
# always did: one re-prefill.
# ---------------------------------------------------------------------------
_state_path: str | None = None
_save_lock = threading.Lock()
# slot -> the _last_end value persist() stamped for a restored pin. It starts IDLE CLEAR's clock; it is NOT a
# request's end, so ONE CONVERSATION's hold ignores it (2026-10-02: after every restart the restored pin "was
# active" at the restart and every new conversation in the next 60 s went to the other card -- the conformance
# check read bonsai-a4000's 138,240 window, and the slots check found both cards held).
_restore_stamp: dict[int, float] = {}


def persist(path: str) -> dict:
    """Load the slot table from `path` and keep it saved there. Returns what
    was restored."""
    global _state_path
    _state_path = path
    restored = {"pins": 0, "prompts": 0, "path": path}
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        return restored
    with _lock:
        for k, s in (st.get("pins") or {}).items():
            if isinstance(s, int) and k not in _pins:
                _pins[k] = s
                _used[k] = float((st.get("used") or {}).get(k) or 0.0)
                _answered.add(k)      # a persisted pin is a conversation that was served (AN UNANSWERED ATTEMPT)
        for s, fp in (st.get("prompts") or {}).items():
            if str(s).isdigit() and isinstance(fp, dict) and fp.get("hashes"):
                _prompts.setdefault(int(s), fp)
        # IDLE CLEAR measures idleness from here for a restored pin: its
        # conversation's last request ended before the restart, unknown when.
        for k, s in _pins.items():
            if s not in _last_end:
                _last_end[s] = _restore_stamp[s] = time.time()
        restored.update(pins=len(_pins), prompts=len(_prompts))
    return restored


def _save() -> None:
    if not _state_path:
        return
    with _lock:
        st = {"pins": dict(_pins), "used": dict(_used),
              "prompts": {str(s): fp for s, fp in _prompts.items()}}
    tmp = _state_path + ".tmp"
    with _save_lock:
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(st, f)
            os.replace(tmp, _state_path)
        except OSError:
            pass


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


# ---------------------------------------------------------------------------
# RANKS: whose cells hold the VRAM line (operator, 2026-09-28).
#
# "Always run from VRAM": the main model's KV pool is tiered (--kv-vram-cells:
# cells below the line in VRAM, the rest in pinned host RAM, read over PCIe on
# every step once a sequence's cells reach past it -- pagoda-h6 decoded 6-15
# tok/s that way). One conversation at the main cap fits below the line; the
# child (64k+) swaps into VRAM while its conversation's main pauses; a second
# concurrent conversation is rare (the relay data: at most 2 at once, only
# Hermes' title call overlapped) and may spill -- "so long as the first
# conversation never spills; if that is not possible, either spilling under
# pressure is fine". The engine decides whose cells move (0041, plan_moves);
# this decides the ranks it reads, and every request carries them:
#
#   kv_rank   the rank of the slot the request runs on
#   kv_ranks  every slot's rank, by slot id -- so the rank of a slot nobody
#             is using is lowered by the next request of anyone
#
# THE LEVELS. RANK_PRIMARY (2): the PRIMARY conversation -- the one that has
# been live longest (a conversation is LIVE while it has a request, a warm or
# a child request in flight, and for PRIMARY_HOLD_S after); primacy passes
# only when the primary stops being live, or to a FORK of it (the proxy's
# `inherits`: pagoda-h6's new session after Hermes compacted on its own was a
# fork of the conversation it continued). RANK_LIVE (1): any other live
# conversation. RANK_NONE (0): a conversation that has left, an empty slot, a
# side call that works for no conversation. The child slot takes the level of
# the conversation its request works for (the request's own conversation,
# bind_request), so the primary's child keeps VRAM while its main is paused
# and the secondary's child spills with it.
#
# PRIMARY_HOLD_S = 60 s, derived from the relay bounds and accepted by the
# operator (2026-09-28): above every gap between two requests of one
# conversation in the Octopus relay logs (482 gaps with no compaction
# between them: median 0.48 s, p99 17.1, max 38.2 s; see IDLE CLEAR) and
# below every gap that spanned a compaction (11, 81 s to 7,984 s). Too long
# costs a second conversation that starts after the first has left at most
# this long spilled; too short lets a newly arrived conversation take the
# first one's VRAM during a long tool step. One conversation at a time is
# the typical use (operator, 2026-09-28).
#
# The switch: YAMADORI_KV_RANK=0 sends no rank fields (the engine then ranks
# every slot alike: 0040's rule with 0041's active flags).
# ---------------------------------------------------------------------------
KV_RANK = os.environ.get("YAMADORI_KV_RANK", "1").strip().lower() not in (
    "0", "", "off", "false", "no")
# PRIMARY_HOLD_S: how long a conversation stays the card's owner after its last request ended (and, under the
# three-slot layout, the primary). The evidence bounds it from BELOW: in the Octopus relay logs (v0b-v0f
# V0-xhigh-1, a conversation's request t_end to its next t0) the 482 gaps with no compaction between them ran
# median 0.48 s, p90 1.73, p99 17.1, max 38.2 s (a `terminal` step) -- one harness, one task family, no human
# think time (the same sample as IDLE_CLEAR_S). A hold above the max keeps every measured in-task gap inside it.
# 60 s is the value in force since 2026-09-28; the stretch from 38.2 to 60 s is not itself measured.
PRIMARY_HOLD_S = float(os.environ.get("YAMADORI_PRIMARY_HOLD_S", "60"))
RANK_PRIMARY, RANK_LIVE, RANK_NONE = 2, 1, 0

# ---------------------------------------------------------------------------
# THE LANE (layout v2, operator 2026-09-29; module docstring "the lane").
#
# RANK_LANE = 3, one above RANK_PRIMARY: engine patch 0041 keeps a
# sequence's cells when its rank is higher than the group claiming the line
# (plan_moves, docs/ENGINES.md "KV rank"), so the lane -- the decider and
# side calls, budget.LANE_TOKENS cells -- never moves out of VRAM and the
# primary never waits on a swap of it. What it costs the main line:
# LANE_TOKENS cells, which budget.main_cap takes off the cap (the deploy
# writes YAMADORI_MAIN_CAP = N - LANE_TOKENS). A second-brain request on the
# lane's slot (while that machinery exists) takes its
# conversation's rank, as the child did. bench/kv_rank.py (layout v2, step
# `lane`) measures both spans and the moves. No switch back (operator,
# 2026-09-29: old decisions are not kept behind switches; git has them).
# LANE_RANK / LANE_KEEP exist for the offline suites only, which turn them off
# to exercise the child-rank rule and the release path that the second brain's
# runs and compactions still take.
# ---------------------------------------------------------------------------
RANK_LANE = RANK_PRIMARY + 1
LANE_RANK = True
# THE LANE IS CLEARED AFTER EACH BURST (operator, 2026-09-29, verbatim: "We clear jjava lane too after it is done
# right, not slow down slop"), reversing layout v2's "kept" rule of the same day. Why: in the unified pool every main
# decode step reads up to the HIGHEST used cell, and the lane's cells can sit above the conversation's top cell (the
# top-cell cost, #59; Flash-Next 8.6 tok/s at 8K beside an idle 64K slot, 2026-09-29). Keeping them saved little:
# the decider's state changes every turn anyway. Within one burst (one state, many questions) the cache is untouched:
# the release comes when decide_turn's Turn closes (decider_bonsai.release -> release_idle, the erase or the
# one-token prompt), and after a side call on the lane (proxy._post_events). Recorded in x_yamadori.slots.released
# with why "lane burst ended" (a decider burst) or "side call ended", and its ms.
LANE_KEEP = False
# The callers' names for the end of a decider burst on the lane, recorded as one reason.
LANE_BURST_WHY = frozenset({"decider turn", "decider batch", "jev call"})


def lane_ranked() -> bool:
    """Is the lane ranked above the primary (and so reserved in VRAM)?"""
    return bool(KV_RANK and LANE_RANK and ENABLED)


def lane_kept() -> bool:
    """Is the lane kept between decider turns and side calls (not released)?"""
    return bool(LANE_KEEP and ENABLED)
_act: dict[str, dict] = {}          # key -> {since, last, inflight}
_primary: str | None = None
_slot_for: dict[int, list] = {}     # slot -> conversation keys of its grants in flight
_helper_on: dict[int, int] = {}     # slot -> second-brain grants in flight (THE LANE)


def _live(k: str | None, now: float) -> bool:
    a = _act.get(k) if k else None
    return bool(a) and (a["inflight"] > 0 or now - a["last"] <= PRIMARY_HOLD_S)


def _touch(k: str | None, now: float, d: int = 0) -> None:
    """Activity of conversation `k`: d=+1 a request of it starts, -1 one
    ends, 0 it arrived. A conversation that was not live starts a new live
    streak (`since`). Called under _lock."""
    if not k or k == HELPER:
        return
    a = _act.get(k)
    if a is None or not _live(k, now):
        a = {"since": now, "last": now,
             "inflight": max(a["inflight"], 0) if a else 0}
        _act[k] = a
    a["last"] = now
    a["inflight"] = max(a["inflight"] + d, 0)
    if len(_act) > 256:
        for old in [x for x, v in _act.items()
                    if v["inflight"] <= 0 and now - v["last"] > 86400]:
            _act.pop(old, None)


def _elect(now: float, by: str | None = None,
           inherits: str | None = None) -> None:
    """Keep the primary while it is live; else the conversation live
    longest. A fork (`by` inherits from the primary `inherits`) takes it
    over. Called under _lock."""
    global _primary
    if by and inherits and inherits == _primary and by != inherits \
            and by in _act:
        _act[by]["since"] = min(_act[by]["since"],
                                (_act.get(inherits) or {}).get("since", now))
        _primary = by
        return
    if _primary is not None and _live(_primary, now):
        return
    live = sorted((a["since"], k) for k, a in _act.items() if _live(k, now))
    _primary = live[0][1] if live else None


def _level(k: str | None, now: float) -> int:
    if not k or k == HELPER:
        return RANK_NONE
    if k == _primary and _live(k, now):
        return RANK_PRIMARY
    return RANK_LIVE if _live(k, now) else RANK_NONE


def _slot_key(s: int) -> str | None:
    """The conversation slot `s` works for now: the newest grant in flight
    on it that names one, else the conversation pinned to it. Called under
    _lock."""
    for k in reversed(_slot_for.get(s) or []):
        if k:
            return k
    for k, p in _pins.items():
        if p == s and k != HELPER:
            return k
    return None


def _ranks(n: int, now: float) -> list[int]:
    """Every slot's rank. THE LANE: RANK_LANE while it is ranked and no
    second-brain request is on it (then its conversation's, as the child's
    was). Called under _lock."""
    lane = n - 1 if n >= 2 and lane_ranked() else None
    return [RANK_LANE if s == lane and not _helper_on.get(s, 0)
            else _level(_slot_key(s), now) for s in range(n)]


def rank_fields(slot: int | None) -> dict:
    """{kv_rank, kv_ranks} for a request on `slot` now ({} when ranks are off
    or there is no slot). For a caller that set id_slot itself (the decider
    through model.post)."""
    if slot is None or not KV_RANK or not ENABLED:
        return {}
    now = time.time()
    with _lock:
        _elect(now)
        ranks = _ranks(max(_known_n(), int(slot) + 1), now)
    return {"kv_rank": ranks[int(slot)], "kv_ranks": ranks}


def upstream_fields(grant: dict | None) -> dict:
    """What a request on `grant`'s slot sends upstream: id_slot and
    cache_prompt, and the ranks (RANKS) when they are on."""
    if not grant or grant.get("slot") is None:
        return {}
    out = {"id_slot": grant["slot"], "cache_prompt": True}
    if grant.get("kv_ranks") is not None:
        out.update(kv_rank=grant["kv_rank"], kv_ranks=list(grant["kv_ranks"]))
    return out


def _grant_ranks(grant: dict, works_for: str | None, now: float,
                 inherits: str | None = None) -> None:
    """Record the request in flight on its slot, elect, and put the ranks on
    the grant. Called under _lock."""
    s = grant.get("slot")
    if s is None:
        return
    grant["_works_for"] = works_for
    _slot_for.setdefault(s, []).append(works_for)
    _touch(works_for, now, +1)
    _elect(now, by=works_for, inherits=inherits)
    if KV_RANK:
        ranks = _ranks(max(_known_n(), s + 1), now)
        grant["kv_rank"] = ranks[s]
        grant["kv_ranks"] = ranks
        grant["primary"] = (_primary or "")[:8] or None


def _works_for(key: str | None) -> str | None:
    """The conversation a request works for: its own key, else the key of
    the request it runs inside (bind_request) -- the second brain's, the
    decider's and a side call's."""
    if key and key != HELPER:
        return key
    ctx = _request_ctx()
    k = (ctx or {}).get("key")
    return k if k and k != HELPER else None


# ---------------------------------------------------------------------------
# ONE CONVERSATION ON THE MAIN CARD (operator, 2026-09-29, verbatim: "we should not have a second conversation at
# all, it is too slow, we have a second gpu if we want a second conversation, that is how it has to play out, the
# jjava engine should be the only other thing we need ready to go"). Evidence: Flash-Next decoded an 8K
# conversation at 8.6 tok/s beside an idle 64K slot -- the unified pool's top-cell cost, as measured on Bonsai (#59).
#
# IN FORCE when the served layout has ONE conversation slot (-np 2: slot 0 the conversation, slot 1 the jjava lane;
# conversation_slots() == [0]). With today's -np 3 nothing below applies, so the proxy can ship before the deploy.
#
# THE OWNER. The card belongs to one conversation: the one pinned to slot 0 while it has a request of this process
# in flight there, or ended one less than PRIMARY_HOLD_S ago. A DIFFERENT conversation arriving then is refused --
# ConversationAtCapacity, HTTP 503 `conversation_at_capacity` with Retry-After (the rest of the hold; while the
# owner is mid-request api_errors' 30 s, the end being unknown) -- until the second Bonsai on the A4000
# (`bonsai-a4000`, not deployed) takes it instead. After the hold the newcomer takes the card and the old owner's
# pin goes (its next request is a new arrival like any other). The lane is never a conversation's.
#
# COMPACTIONS ARE NEVER REFUSED FOR BEING COMPACTIONS (coordinator, 2026-09-29: a refused or cut summary makes
# Hermes discard it and compact again -- the h6 failure):
#   the OWNER's compaction (its key: in place, or a flattened one mapped to it)  -> slot 0, its own cache;
#   one sent up as is with no conversation identity  -> slot 0 once slot 0 has no request in flight, waiting for
#       that like any request, at most COMPACTION_WAIT_S, then the Retry-After path;
#   a DIFFERENT conversation's (keyed)  -> the second-conversation rule above.
#   NEVER the lane (3,072 cells: budget.LANE_TOKENS).
# ---------------------------------------------------------------------------
_slot_free = threading.Condition(_lock)
# The as-sent compaction's wait for slot 0: admission's own bound for a request waiting for room
# (admission.WAIT_SECONDS, YAMADORI_ADMIT_WAIT, 20 s), not a number of this rule's own.
COMPACTION_WAIT_S = float(os.environ.get("YAMADORI_ADMIT_WAIT", "20"))
RETRY_AFTER_UNKNOWN = 30          # api_errors' existing 503 Retry-After (of_upstream), the end not known


class ConversationAtCapacity(RuntimeError):
    """Another conversation owns the main card (ONE CONVERSATION). Retryable: 503 conversation_at_capacity."""

    def __init__(self, why: str, retry_after: int, owner: str | None = None):
        super().__init__(why)
        self.retryable = True
        self.retry_after = int(retry_after)
        self.owner = owner


def one_conversation(n: int | None = None) -> bool:
    """The served layout has exactly one conversation slot: -np 2 (the conversation + the lane) or -np 1 (a LOCKED
    card, operator 2026-09-30: the conversation alone; jjava and side calls on bonsai-a4000)."""
    n = count() if n is None else n
    return ENABLED and n in (1, 2)


def _helper_server_grant(transient: bool, key: str | None) -> dict | None:
    """A request that runs on ANOTHER server than the main card (the tier table's helper, `bonsai-a4000`: jjava and
    side calls of a locked tier) gets no slot from this registry, which counts the main card's slots: the helper
    server picks its own (its -np 2, by prompt similarity). None when the request is for the main card.
    The target: the request's own model when it is not a main one (a client side call the tier table routed to the
    helper: server.py binds it, max_mode.set_current), else jjava's (max_mode.decider_model: a transient request
    with no compaction prefix, made while a main model is bound, is a decider read). A compaction never reaches
    here (it carries `prefix`, or is keyed to its conversation: proxy.prepare `_slot`). The table's rows name ONE
    helper for both (helpers.decider == helpers.side_calls; mcp/test_max_mode.py checks it)."""
    if not transient or key not in (None, HELPER):
        return None
    try:
        import max_mode
        if not max_mode.ENABLED:
            return None
        target = max_mode.current()
        if not target or max_mode.is_main(target):
            target = max_mode.decider_model()
        if not target or max_mode.is_main(target):
            return None
    except Exception:                                                # noqa: BLE001
        return None
    # its LANE slot (OTHER_LANE_SLOT): slot 0 there is the other card's conversation (THE OTHER CARD)
    return {"slot": OTHER_LANE_SLOT, "mode": "helper server", "evicted": None, "_server": target,
            "how": f"{target}: its lane (slot {OTHER_LANE_SLOT}; slot {OTHER_CONV_SLOT} is the other card's "
                   "conversation)"}


# THE LIVE SUITE'S HOLD (X-Yamadori-Features {"primary_hold_s": N}; honoured by proxy.prepare for a TEST account
# only, as IDLE CLEAR's `idle_clear_s`): the live suite opens a new conversation every few seconds, and inside the
# 60 s hold each one was routed to the other card or told 503 -- deploy check 2026-10-01, 12 failures, the
# conformance window read as bonsai-a4000's 138,240. The override shortens the hold only AGAINST an owner of the
# SAME test account (its own conversations); a request in flight still holds the card, and a client's owner is held
# for the full PRIMARY_HOLD_S whatever a test sends. key -> (hold seconds, account) for its current request.
_hold_for: dict[str, tuple[float, str]] = {}


def _hold_against(owner: str | None, key: str | None) -> float:
    """The hold `owner` has against the newcomer `key`: PRIMARY_HOLD_S, or the newcomer's test override when both
    conversations are the same (test) account's. Under _lock."""
    h = _hold_for.get(key) if key else None
    if h and owner and _key_account.get(owner) == h[1]:
        return h[0]
    return PRIMARY_HOLD_S


def _owner_of_card(now: float, key: str | None = None) -> tuple[str | None, bool, float]:
    """(the owner's key or None, it has a request in flight, seconds since its LATEST ACTIVITY -- its last request's
    end or its latest arrival, so its queued follow-ups keep the card: operator 2026-09-30), as seen by the newcomer
    `key` (_hold_against). Under _lock."""
    k = next((k for k, s in _pins.items() if s == 0 and k != HELPER), None)
    if k is None:
        return None, False, 0.0
    busy = _busy.get(0, 0) > _warming.get(0, 0)
    ended = _last_end.get(0, 0.0)
    if _restore_stamp.get(0) == ended:
        ended = 0.0                       # a restart's stamp on a restored pin, not a request's end
    since = now - max(ended, _used.get(k, 0.0), _last_seen.get(k, 0.0))
    if busy or since < _hold_against(k, key):
        return k, busy, since
    return None, False, since


def _other_owner(now: float, key: str | None = None) -> tuple[str | None, bool, float]:
    """The other card's (owner, busy, since latest activity), by the same hold rule. Under _lock."""
    k = _other["owner"]
    if k is None:
        return None, False, 0.0
    busy = _other["busy"] > 0
    since = now - max(_other["last_end"], _last_seen.get(k, 0.0))
    if busy or since < _hold_against(k, key):
        return k, busy, since
    return None, False, since


def _rest_of_hold(busy: bool, since: float, hold: float | None = None) -> int:
    hold = PRIMARY_HOLD_S if hold is None else hold
    return RETRY_AFTER_UNKNOWN if busy else max(1, int(hold - since + 0.999))


def other_card_for(model: str | None) -> tuple[str | None, str]:
    """(the other card's model for a request served by `model`, why): the tier table's `other_card`
    (max_mode.other_card). None when the table is off or its model cannot run there (flash-next, mirai-s) -- unless
    the one switch YAMADORI_OTHER_CARD_DOWNGRADE=1 lets it run on the default model's other card."""
    try:
        import max_mode
        # A request max_mode ALREADY sent to the other card's model (a lower tier while a higher tier's model holds the
        # main card: operator 2026-10-06, "Serve it on the A4000") runs there itself.
        if model and max_mode.ENABLED and not max_mode.is_main(model) and max_mode.TABLE.full and any(
                (max_mode.TABLE.row(m).get("other_card") == model) for m in max_mode.MODELS):
            return model, f"{model} is the other card's model (max_mode routed this request there)"
        return max_mode.other_card(model)
    except Exception as e:                                           # noqa: BLE001
        return None, f"no tier table ({type(e).__name__})"


def _route(key: str, model: str | None, now: float) -> dict | None:
    """ONE CONVERSATION PER CARD, with the OTHER card. None: serve on the main card (its owner, or it is free).
    A dict: route to the other card. Raises ConversationAtCapacity when neither card may take it. Under _lock."""
    if _other["owner"] == key:
        om, _why = other_card_for(model)
        if om and om == _other["model"]:
            # it keeps that card for its life
            return {"routed": "other_card", "model": om, "why": "this conversation's card (it keeps it for its life)"}
        _other.update(owner=None, model=None)          # its tier's model cannot run there now: it is a newcomer again
    owner, busy, since = _owner_of_card(now, key)
    try:
        import max_mode
        sent_there = bool(model and max_mode.ENABLED and not max_mode.is_main(model)
                          and other_card_for(model)[0] == model)
    except Exception:                                                # noqa: BLE001
        sent_there = False
    if sent_there:
        # max_mode already sent it to the other card's model (operator 2026-10-06): it takes that card whoever owns
        # the main one, unless another conversation holds the other card
        o, obusy, osince = _other_owner(now, key)
        if o is not None and o != key:
            raise ConversationAtCapacity(
                f"this request's tier runs on {model} while a higher tier's model holds the main card, and {model} "
                f"holds another conversation ({'in flight' if obusy else f'active {osince:.0f} s ago'}); one "
                "conversation per card (operator, 2026-09-30)", _rest_of_hold(obusy, osince, _hold_against(o, key)),
                o)
        _other.update(owner=key, model=model)
        return {"routed": "other_card", "model": model,
                "why": f"a higher tier's model holds the main card: this tier runs on {model} (operator 2026-10-06)"}
    if owner is None or owner == key:
        return None
    hold = _hold_against(owner, key)
    om, why_not = other_card_for(model)
    if not om:
        raise ConversationAtCapacity(
            f"another conversation holds the main card ({'a request in flight' if busy else f'its latest activity was {since:.0f} s ago, the hold is {hold:.0f} s'}), "
            f"and this request's model ({model}) does not run on the other card ({why_not}); one conversation per card "
            "(operator, 2026-09-30)", _rest_of_hold(busy, since, hold), owner)
    o, obusy, osince = _other_owner(now, key)
    if o is not None and o != key:
        raise ConversationAtCapacity(
            f"both cards hold a conversation: the main card's ({'in flight' if busy else f'active {since:.0f} s ago'}) "
            f"and {om}'s ({'in flight' if obusy else f'active {osince:.0f} s ago'}); one conversation per card "
            "(operator, 2026-09-30)", min(_rest_of_hold(busy, since, hold),
                                          _rest_of_hold(obusy, osince, _hold_against(o, key))), owner)
    rec = {"routed": "other_card", "model": om, "owner": owner[:8], "owner_busy": busy,
           "owner_idle_s": round(since, 1),
           "why": (f"the main card's owner ({owner[:8]}) " + ("has a request in flight" if busy else
                   f"was active {since:.0f} s ago, inside the {hold:.0f} s hold") +
                   f": a new conversation id takes the other card ({om}), and keeps it")}
    prev = _other["owner"]
    if prev and prev != key:
        rec["took_from"] = prev[:8]
        _displaced[prev] = {"at": now, "card": om, "by": key[:8]}
    _other.update(owner=key, model=om)
    return rec


def _refuse_second(key: str, now: float) -> None:
    """Raise ConversationAtCapacity when another conversation owns the main card (acquire's backstop: check_owner
    routed it first; this is the race where the card was taken between the two). Under _lock."""
    owner, busy, since = _owner_of_card(now, key)
    if owner is None or owner == key:
        return None
    hold = _hold_against(owner, key)
    ra = _rest_of_hold(busy, since, hold)
    raise ConversationAtCapacity(
        f"another conversation holds the main card ({'a request in flight' if busy else f'its last request ended {since:.0f} s ago, the hold is {hold:.0f} s'}); "
        f"one conversation per card (operator, 2026-09-29)", ra, owner)


def check_owner(want: dict | None, model: str | None = None) -> dict | None:
    """ONE CONVERSATION PER CARD, decided EARLY (proxy._run_turn right after prepare, before any generation), for
    the request's conversation (`want` = prepare's `_slot`) served by `model`:
      None     the main card (its owner; or the card is free -- a new id takes it after the hold: want["switch"])
      a dict   the OTHER card ({routed, model, why, owner_idle_s, ...}; also want["routed"]): the proxy sends this
               request, and every later one of the conversation, to that model
      raises   ConversationAtCapacity (503 + Retry-After): neither card may take it -- the other card is held, or
               its model cannot run there (flash-next, mirai-s; the one switch YAMADORI_OTHER_CARD_DOWNGRADE)
    A no-op off one conversation, for a side call and for a compaction sent up as is (never refused for being one);
    a compaction or a warm WITH its conversation's key goes to that conversation's own card. Its arrival is
    activity: the hold counts from an owner's latest arrival, so its queued follow-ups keep the card, and when the
    card frees the owner's request is served first (only the owner's requests are ever granted the main card
    while its hold runs)."""
    if not want or not one_conversation():
        return None
    key = want.get("key")
    if not key or key == HELPER or want.get("transient"):
        return None
    account = want.get("account") or None
    now = time.time()
    with _lock:
        if account:
            _key_account[key] = account
        # the live suite's hold override for this request (proxy.prepare sets it for a TEST account only)
        if want.get("hold_s") is not None and account:
            _hold_for[key] = (max(0.0, float(want["hold_s"])), account)
        else:
            _hold_for.pop(key, None)
        r = _route(key, model, now)
        if r is None:
            owner = next((k for k, s in _pins.items() if s == 0 and k not in (HELPER, key)), None)
            if owner is not None:
                # after the hold a new id takes the main card: the switch (acquire makes it)
                o_since = now - max(_last_end.get(0, 0.0), _used.get(owner, 0.0), _last_seen.get(owner, 0.0))
                want["switch"] = {"from": owner[:8], "owner_idle_s": round(o_since, 1), "saved": SWITCH_SAVE}
        _last_seen[key] = now
    if r is not None:
        want["routed"] = dict(r)
    return r


def _wait_slot0_free(deadline: float) -> bool:
    """Wait (under _lock, releasing it while waiting) until slot 0 has no request in flight. True when free."""
    import cancel
    while _busy.get(0, 0) > 0:
        left = deadline - time.time()
        if left <= 0:
            return False
        cancel.check()
        _slot_free.wait(min(left, 0.5))
    return True


def _acquire_one(key: str | None, transient: bool, prefix: dict | None, n: int) -> dict:
    """acquire() under ONE CONVERSATION (-np 2). Conversations and compactions on slot 0; everything else keeps
    the lane (the caller's own path)."""
    now = time.time()
    compaction = bool(prefix) and (transient or not key)
    with _lock:
        if compaction:
            if not _wait_slot0_free(now + COMPACTION_WAIT_S):
                raise ConversationAtCapacity(
                    f"a compaction sent as is waited {COMPACTION_WAIT_S:.0f} s for the main card's conversation slot "
                    "(a request in flight there)", RETRY_AFTER_UNKNOWN)
            mode, how = "compaction", ("compaction sent as is: the one conversation slot (never the "
                                                "lane)")
            owner = next((k for k, s in _pins.items() if s == 0 and k != HELPER), None)
            displaced = owner
        else:
            _refuse_second(key, now)
            prev = next((k for k, s in _pins.items() if s == 0 and k not in (HELPER, key)), None)
            switch = None
            if prev is not None:
                p_since = now - max(_last_end.get(0, 0.0), _used.get(prev, 0.0), _last_seen.get(prev, 0.0))
                _pins.pop(prev, None)
                _used.pop(prev, None)
                # THE SWITCH (operator 2026-09-30): the previous owner's state goes to llama-server's host-RAM prompt
                # cache (--cache-ram) when this prompt diverges from it, and comes back if it returns
                _displaced[prev] = {"at": now, "card": "main", "by": key[:8]}
                switch = {"from": prev[:8], "owner_idle_s": round(p_since, 1), "saved": SWITCH_SAVE}
            back = _displaced.pop(key, None)
            _pins[key] = 0
            _used[key] = now
            mode, displaced = "pinned", None
            how = ("pinned: the one conversation slot" + (
                f" (the card passed from {prev[:8]}, idle past the hold)" if prev else ""))
        _cleared.pop(0, None)
        _busy[0] = _busy.get(0, 0) + 1
        _prompts.pop(0, None)
        grant = {"slot": 0, "mode": mode, "how": how, "evicted": None, "one_conversation": True}
        if mode == "pinned":
            grant["_key"] = key           # AN UNANSWERED ATTEMPT HOLDS NOTHING (remember / release)
        if mode == "pinned" and switch:
            grant["switch"] = switch
        if mode == "pinned" and back and back.get("card") == "main":
            # it comes back after a switch: restored from the server's host-RAM prompt cache, or re-processed --
            # cache_record says which, and in how many ms (resumed_how)
            grant["resumed_cold"] = {"slot": 0, "cleared_at": round(back["at"], 1),
                                     "idle_s": round(now - back["at"], 1), "cells_cleared": None,
                                     "by": f"a switch: {back['by']} took the card after the hold"}
        _grant_ranks(grant, key if mode == "pinned" else _works_for(None), now)
    if compaction:
        global _compacting
        with _lock:
            _compacting += 1
        grant["compaction"] = True
        if displaced:
            grant["displaced"] = displaced[:8]
            grant["_displaced_key"] = displaced
    _save()
    return grant


def acquire(key: str | None, transient: bool = False,
            prefix: dict | None = None, warm: bool = False,
            prompt: dict | None = None,
            inherits: str | None = None) -> dict:
    """Choose a slot and count a request in flight on it.

    Returns {slot, mode, how, evicted}: `slot` is None when every slot is busy
    (the server then picks, as it did before pinning). Always pair with
    release(grant).

    `prefix` (a fingerprint(), transient requests only): try COMPACTION
    AFFINITY first. The grant then carries `affinity` -- {key (8 chars),
    slot, shared_tokens, candidates, took, min_tokens} -- whether or not a
    slot was taken, so the record says why it went where it went.

    `prompt` (a fingerprint() of this request, conversation keys): a key
    with no pin ADOPTS the idle slot whose prompt this one continues (see
    ADOPTION); the grant then carries `adopted` {key (8 chars), slot,
    shared_tokens, why}.

    `inherits` (a conversation key): this conversation is a FORK of that one
    and takes over its primacy when it holds it (RANKS)."""
    if not ENABLED:
        return {"slot": None, "mode": "off", "how": "pinning disabled",
                "evicted": None}
    if key and key != HELPER and not transient and _other["owner"] == key:
        # THE OTHER CARD: this conversation's slot there (its own server; nothing of this registry's main slots)
        with _lock:
            if warm:
                _other["warming"] += 1
            else:
                _other["busy"] += 1
            return {"slot": OTHER_CONV_SLOT, "mode": "warm" if warm else "other card", "evicted": None,
                    "_server": _other["model"], "warm": bool(warm), "card": "other",
                    "how": f"{_other['model']}: the other card's conversation slot ({OTHER_CONV_SLOT})"}
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
            grant = {"slot": slot, "mode": "warm", "how": "warm: this "
                     "conversation's slot", "evicted": None, "warm": True}
            _grant_ranks(grant, key, time.time())
        return grant
    hs_grant = _helper_server_grant(transient, key) if not prefix else None
    if hs_grant is not None:
        return hs_grant
    n = count()
    if one_conversation(n) and ((key and key != HELPER and not transient)
                                or (prefix and (transient or not key))):
        # ONE CONVERSATION (above): a conversation's turn or a compaction -> slot 0, or refused
        return _acquire_one(key, transient, prefix, n)
    keep = _transient_slot(n)
    hs = helper_slot(n)
    now = time.time()
    evicted = None
    aff = None
    adopted = None
    adopt_key = None
    cold = None
    if (key == HELPER and not transient) or (
            (transient or not key) and not prefix):
        # THE CHILD SLOT (THE RULE): the second brain, the decider, a side
        # call -- busy or not (the server defers a request behind the one in
        # flight there: a wait, never a conversation's cache), and it never
        # takes or evicts a pin. With one slot the server chooses.
        slot = hs
        works_for = _works_for(None if key == HELPER else key)
        with _lock:
            ev = _releasing.get(slot) if slot is not None else None
            queued = slot is not None and _busy.get(slot, 0) > 0
            mode = "helper" if key == HELPER and not transient else "transient"
            if slot is not None:
                _busy[slot] = _busy.get(slot, 0) + 1
                _prompts.pop(slot, None)
                if mode == "helper":
                    # THE LANE: a second-brain request takes its
                    # conversation's rank while it is on the lane's slot.
                    _helper_on[slot] = _helper_on.get(slot, 0) + 1
            grant = {"slot": slot, "mode": mode,
                     "how": (("helper: the child slot" if mode == "helper"
                              else "transient: the lane (the child slot)")
                             + (" (queued behind the request in flight there)"
                                if queued else "")
                             if slot is not None else
                             "one slot: the server chooses"),
                     "evicted": None}
            if mode == "helper" and slot is not None:
                grant["_helper_on"] = True
            _grant_ranks(grant, works_for, now)
        _save()
        _await_release(ev, grant)
        return grant
    stray = []
    displaced = None
    with _lock:
        # A PIN TO THE CHILD SLOT (or past the end) is dropped: pins persist
        # across restarts (`persist`), and a pin taken while the server ran
        # four slots can name slot 2 -- the child's of three. Seen
        # 2026-09-29 in logs/proxy.log ("slot 2 (decider turn) kept -- a
        # conversation's pinned slot"): a conversation on the child slot
        # shares it with the decider and blocks its release.
        convo = set(conversation_slots(n))
        for k2 in [k2 for k2, s2 in _pins.items()
                   if k2 != HELPER and s2 not in convo]:
            stray.append((k2, _pins.pop(k2)))
            _used.pop(k2, None)
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
            # A compaction sent up as is that shares no conversation's
            # prefix. LAYOUT V2: the lane holds LANE_TOKENS cells and this
            # holds its whole transcript, so it goes to the least recently
            # used CONVERSATION slot (coordinator, 2026-09-29: a cache miss
            # beats a failed compaction), which is emptied after it
            # (proxy._post_events) and whose conversation's next request is
            # reported as a cold resume.
            slot, how, displaced = _compaction_slot(n, busy)
            mode = "compaction"
        else:
            mine = _pins.get(key)
            only_warm = (mine is not None and mine in busy
                         and _warming.get(mine, 0) >= _busy.get(mine, 0))
            adopt = (_adopt(prompt, busy) if mine is None and prompt
                     else None)
            if adopt is not None:
                # ADOPTION (above): the slot holding this conversation's
                # prefix under its old key; the pin moves to this key.
                slot = adopt["slot"]
                _pins.pop(adopt["key"], None)
                _used.pop(adopt["key"], None)
                _pins[key] = slot
                adopt_key = adopt["key"]
                adopted = dict(adopt, key=adopt["key"][:8])
                how = (f"adopted: this prompt continues the one on slot "
                       f"{slot} (key {adopt['key'][:8]}, ~"
                       f"{adopt['shared_tokens']} tokens, {adopt['why']}); "
                       f"the pin moves to this key")
            elif mine is not None and (mine not in busy or only_warm):
                slot, how = mine, ("pinned: this conversation's slot"
                                   + (" (after its warm)" if only_warm else ""))
                cold = _cleared.pop(slot, None)
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
        ev = None
        if slot is not None:
            ev = _releasing.get(slot)
            if mode != "pinned" or cold is None:
                # Anyone else on a cleared slot: it is no longer the
                # conversation's cold slot to report (its pin moved).
                _cleared.pop(slot, None)
            _busy[slot] = _busy.get(slot, 0) + 1
            # Its content is about to change; remember() writes it back.
            _prompts.pop(slot, None)
        grant = {"slot": slot, "mode": mode, "how": how,
                 "evicted": (evicted or "")[:8] or None}
        # RANKS: a conversation's own turn works for it; a compaction sent up
        # as is for the conversation of the request it runs inside, if any.
        # An adopted pin is the same conversation under a new key: it takes
        # over the old key's primacy, as a fork does (`inherits`).
        _grant_ranks(grant, key if mode == "pinned" else _works_for(None),
                     now, inherits=(inherits or (adopt_key if adopted
                                                 else None)))
    if aff is not None:
        grant["affinity"] = dict(aff, key=(aff["key"] or "")[:8] or None)
    if adopted is not None:
        grant["adopted"] = adopted
    if cold is not None:
        # IDLE CLEAR emptied this conversation's slot while it was away:
        # this request re-processes its prompt, and says so.
        grant["resumed_cold"] = dict(cold, slot=slot)
    if prefix is not None and (transient or not key):
        # A compaction sent up as is: attributed to no conversation, so no
        # idle slot is cleared while it runs (IDLE CLEAR).
        global _compacting
        with _lock:
            _compacting += 1
        grant["compaction"] = True
    if displaced is not None:
        grant["displaced"] = displaced[:8]
        grant["_displaced_key"] = displaced
    if stray:
        grant["stray_pins_dropped"] = [{"key": k[:8], "slot": s2}
                                       for k, s2 in stray]
    _save()
    _await_release(ev, grant)
    return grant


def _compaction_slot(n: int, busy: set) -> tuple[int | None, str, str | None]:
    """(slot, how, the conversation whose cache it displaces) for a
    compaction sent up as is (layout v2): a conversation slot no one is
    pinned to, else the least recently used idle pinned one (its pin is
    kept; its cache is not), else the least recently used busy one (queued
    there). Called under _lock."""
    convo = conversation_slots(n)
    held = {s: k for k, s in _pins.items() if k != HELPER and s in convo}
    free = [s for s in convo if s not in held and s not in busy]
    if free:
        return free[0], ("compaction sent as is: a conversation slot no "
                         "conversation is pinned to"), None
    for want_idle in (True, False):
        c = sorted((_used.get(k, 0.0), s, k) for s, k in held.items()
                   if (s not in busy) == want_idle)
        if c:
            _t, s, k = c[0]
            return s, ("compaction sent as is: the least recently used "
                       f"conversation slot ({k[:8]}'s; its pin is kept, its "
                       "cache is not)"
                       + ("" if want_idle else
                          " (queued behind the request in flight there)")), k
    return None, "compaction sent as is: no conversation slot; the server " \
                 "chooses", None


def release(grant: dict | None) -> None:
    global _compacting
    if grant and grant.get("compaction"):
        with _lock:
            _compacting = max(_compacting - 1, 0)
        grant["compaction"] = False
    if not grant or grant.get("slot") is None:
        return
    if grant.get("_server"):
        # another server's slot (THE OTHER CARD, or a helper server's lane): nothing of this registry's
        if grant.get("card") == "other" and not grant.get("_released"):
            with _lock:
                grant["_released"] = True
                if grant.get("warm"):
                    _other["warming"] = max(_other["warming"] - 1, 0)
                else:
                    _other["busy"] = max(_other["busy"] - 1, 0)
                    _other["last_end"] = time.time()
        return
    with _lock:
        s = grant["slot"]
        _busy[s] = max(_busy.get(s, 0) - 1, 0)
        k_own = grant.get("_key")
        if (k_own and grant.get("one_conversation") and not grant.get("warm") and not grant.get("_answered")
                and k_own not in _answered and _pins.get(k_own) == s and not _busy.get(s, 0)):
            # AN UNANSWERED ATTEMPT HOLDS NOTHING: this conversation has never been answered and this request
            # ended without a generation -- its pin and its activity go, and the slot's end is not stamped, so a
            # retry (a new id) finds the card free
            _pins.pop(k_own, None)
            _used.pop(k_own, None)
            _last_seen.pop(k_own, None)
            grant["unanswered"] = True
        else:
            _last_end[s] = time.time()
        _slot_free.notify_all()           # ONE CONVERSATION: a compaction may be waiting for slot 0
        if grant.get("warm"):
            _warming[s] = max(_warming.get(s, 0) - 1, 0)
        if grant.get("mode") in ("helper", "transient", "compaction"):
            # This process filled it with cache no conversation owns.
            _dirty.add(s)
        if grant.pop("_helper_on", False):
            _helper_on[s] = max(_helper_on.get(s, 0) - 1, 0)
        k_disp = grant.pop("_displaced_key", None)
        if k_disp and _pins.get(k_disp) == s:
            # Its conversation's next request re-processes its prompt, and
            # says so (x_yamadori.slots.resumed_cold), as after IDLE CLEAR.
            _cleared[s] = {"cleared_at": round(time.time(), 1),
                           "idle_s": None, "cells_cleared": None,
                           "by": "a compaction sent as is"}
        if "_works_for" in grant:
            # RANKS: this request no longer works for its conversation.
            wf = grant.pop("_works_for")
            held_for = _slot_for.get(s) or []
            if wf in held_for:
                held_for.remove(wf)
            _touch(wf, time.time(), -1)


# ---------------------------------------------------------------------------
# RELEASE: empty the slots whose cache nobody will reuse.
#
# THE COST (module docstring, "AN IDLE SLOT'S CACHE IS NOT FREE"): every cell
# any slot holds is attended over by every other slot's decode. A linear fit
# of the kv_share_probe's six cells (n=2 each; inferred, not a separate
# measurement): ~16 ms per decoded token plus ~0.27 ms per 1,000 cells of
# n_kv -- the pool up to its HIGHEST used cell, empty or masked cells below
# it included (2026-09-27, #59: a leftover ABOVE the active conversation
# costs ~11 ms a token at 32k and clearing it gives that back; one BELOW
# costs as much and clearing it gives back little, because the active
# conversation's own cells keep the top where it was).
#
# WHAT IS RELEASED, AND WHEN (operator brief 2026-09-26; each a CHOICE):
#   the second brain's slot   after each second-brain RUN: when the helper
#                             lane is let go (admission.helper_lane), which
#                             is the end of a shomen.run job -- investigate,
#                             plan, fixup, alternative, tiebreak -- or of a
#                             fan-out's B AND C together (fanout.run holds
#                             one lane across both). Never between the hops
#                             of a job: they reuse each other's prefix.
#                             Never between B and C: C's prompt is B's with
#                             the last user turn replaced, and the server
#                             keeps a checkpoint at the start of that turn
#                             (server-context.cpp "break at the last user
#                             message"), so C reuses B's whole history.
#   the transient slot        after each client side call (and a compaction
#                             sent up as is) that ran on it. Nothing reuses
#                             it: COMPACTION AFFINITY only ever considers
#                             conversations' PINNED slots (_affinity), a
#                             compaction's continuation opens with the tools
#                             and the summary, never the flattened transcript,
#                             and side calls do not share a prefix worth
#                             keeping (a retried compaction does: it pays one
#                             re-prefill).
#   never                     a slot a conversation is pinned to (a compaction
#                             placed there by affinity included), a slot with
#                             a request of this process in flight, the
#                             helper's slot while a second-brain run holds a
#                             helper lane, a slot another process is
#                             generating on (read from /slots is_processing),
#                             or a slot this process did not fill (_dirty):
#                             nothing to gain, and no request to a model
#                             server nobody asked for.
#
# ONE PROCESS RELEASES: the proxy, which calls enable_release() at startup
# (server.py), where every shomen.run, fan-out and side call runs. The worker
# and the tools API use the helper slot too, from their own processes, and
# cannot see a proxy job between two hops (the slot is idle server-side while
# a tool runs); a release from there would wipe the job's prefix. Their
# leftovers are released by the proxy's next second-brain run.
#
# HOW (model.release_slot, the one door): llama-server's own erase
# (POST /slots/<id>?action=erase, server-context.cpp SLOT_ERASE: prompt_clear,
# seq_rm of the whole sequence) when the server allows it -- it answers 501
# unless started with --slot-save-path, which config.yaml does not set -- and
# otherwise the kv_share_probe's method: a one-token prompt of its own on that
# slot (/completion, id_slot, cache_prompt, n_predict 0), which diverges at
# position 0 and so clears the slot. The switch: X-Yamadori-Features
# {"slot_release": false} or YAMADORI_SLOT_RELEASE=0 (tiers.BEHAVIOURS;
# default on); x_yamadori.slots.released records each one.
# ---------------------------------------------------------------------------
RELEASE_WAIT_S = float(os.environ.get("YAMADORI_SLOT_RELEASE_WAIT", "15"))
_release_enabled = False
# cancel.Token -> {"on", "source", "log"}: the request a release belongs to.
_requests: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def enable_release(on: bool = True) -> None:
    """This process releases slots (server.py, the proxy, at startup)."""
    global _release_enabled
    _release_enabled = bool(on)


def release_enabled() -> bool:
    return _release_enabled


def bind_request(token, on: bool, source: str, log: list | None, *,
                 key: str | None = None, idle_on: bool | None = None,
                 idle_log: list | None = None, account: str | None = None,
                 idle_after_s: float | None = None) -> None:
    """The request `token` belongs to: its RELEASE switch (tiers.
    behaviour_source "slot_release") and the list x_yamadori.slots.released
    reads; its IDLE CLEAR switch and list (cleared_idle); and its
    conversation `key`, which IDLE CLEAR never clears for it and counts as
    arrived now. A job thread bound to the same token (cancel.bound) finds
    it."""
    if key:
        with _lock:
            _last_seen[key] = time.time()
            if account:
                _key_account[key] = account
            # RANKS: an arrival is activity (a live streak starts or goes on).
            _touch(key, _last_seen[key])
    if token is not None:
        try:
            _requests[token] = {"on": bool(on), "source": source, "log": log,
                                "key": key, "idle_on": idle_on,
                                "idle_log": idle_log, "account": account,
                                "idle_after_s": idle_after_s}
        except TypeError:
            pass


def _request_ctx() -> dict | None:
    try:
        import cancel
        tok = cancel.current()
    except Exception:                                            # noqa: BLE001
        return None
    try:
        return _requests.get(tok) if tok is not None else None
    except TypeError:
        return None


def _await_release(ev, grant: dict) -> None:
    """A release in flight on the granted slot: wait for it (bounded) so this
    request is sent after it, never overwritten by it."""
    if ev is None:
        return
    t0 = time.time()
    ev.wait(RELEASE_WAIT_S)
    grant["waited_release_ms"] = round((time.time() - t0) * 1000)


def release_idle(slot: int | None, why: str, *, model: str | None = None,
                 on: bool | None = None, log: list | None = None,
                 pinned_ok: bool = False) -> dict | None:
    """Empty `slot` now if the rules above allow it. Returns the record (also
    appended to `log`, else to the current request's), or None when nothing
    was attempted: not this process's job, switched off, or nothing to
    release. Never raises. `pinned_ok`: the slot may be a conversation's pin
    (layout v2: a compaction sent as is displaced that conversation's cache,
    so nothing of it is left to keep). Never raises."""
    if slot is None or not ENABLED or not _release_enabled:
        return None
    ctx = _request_ctx()
    if on is None:
        on = ctx["on"] if ctx else _release_default()
    if not on:
        return None
    if log is None and ctx:
        log = ctx.get("log")
    rec = {"slot": slot, "why": why}
    if why in LANE_BURST_WHY and slot == child_slot(_known_n()):
        rec.update(why="lane burst ended", by=why)
    with _lock:
        if slot not in _dirty:
            return None
        reason = None
        if slot in _pins.values() and not pinned_ok:
            reason = "a conversation's pinned slot"
        elif _busy.get(slot, 0) > 0:
            reason = "a request of this process is on it"
        elif slot in _releasing:
            reason = "already being released"
        elif slot == helper_slot(_known_n()) and _helper_run_active():
            reason = "a second-brain run holds a helper lane"
        if reason is None:
            ev = threading.Event()
            _releasing[slot] = ev
            _busy[slot] = _busy.get(slot, 0) + 1
            _dirty.discard(slot)
            _prompts.pop(slot, None)
    if reason is not None:
        rec.update(released=False, skipped=reason)
        _note(rec, log)
        return rec
    res = _send_release(slot, ev, model)
    _fill(rec, res)
    if res.get("skipped") and "processing" in str(res["skipped"]):
        with _lock:
            # Another process is using it: it may still hold our leftovers.
            _dirty.add(slot)
    _save()
    _note(rec, log)
    return rec


def _send_release(slot: int, ev, model: str | None) -> dict:
    """model.release_slot on a slot already marked busy and releasing (under
    _lock by the caller); unmark it and wake any waiter. Never raises."""
    try:
        import model as _model
        res = _model.release_slot(slot, **({"model": model} if model else {}))
    except Exception as e:                                       # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    finally:
        with _lock:
            _busy[slot] = max(_busy.get(slot, 0) - 1, 0)
            _releasing.pop(slot, None)
        ev.set()
    return res


def _fill(rec: dict, res: dict) -> None:
    rec.update(released=bool(res.get("ok")) and not res.get("skipped"),
               cells_before=res.get("cells_before"), ms=res.get("ms"),
               method=res.get("method"))
    for k in ("skipped", "error", "erase_refused"):
        if res.get(k):
            rec[k] = res[k]


# ---------------------------------------------------------------------------
# IDLE CLEAR: an idle CONVERSATION's slot, cleared for an active request.
#
# (operator, 2026-09-26: "If we get big token boost then clear the idle, no
# brainer"; #59 in docs/SELF-IMPROVEMENT-LOG.md.) The same cost as RELEASE
# above -- every occupied cell slows every other slot's decode -- but the
# cache is a conversation's, so it goes only when the conversation has
# plainly left, and only for a request that is about to generate.
#
# THE RULE. When a request is about to generate on slot X (clear_idle, from
# proxy._post_events and model.post, after acquire), every OTHER
# conversation's pinned slot that has been idle for more than IDLE_CLEAR_S is
# cleared. Idle: no request or warm of this process in flight on it, no warm
# pending for its conversation (idle_guard, set by the proxy), measured from
# the end of its last request or warm (_last_end) and from its conversation's
# last arrival (_last_seen, bind_request). Never the requesting
# conversation's own slot (a lone conversation never clears itself; a
# second-brain run or side call working for a conversation never clears
# that conversation), never while a compaction sent up as is is in flight
# (it names no conversation; a mapped or in-place compaction runs on its
# conversation's slot, which is then busy), never a slot another process is
# generating on (/slots is_processing). The PIN IS KEPT and the ledger is
# untouched: the conversation keeps its slot and replays byte for byte; only
# the cells go. Its next request re-processes the prompt and its record says
# so (x_yamadori.slots.resumed_cold); the clearing request's record lists
# what it cleared (x_yamadori.slots.cleared_idle).
#
# IDLE_CLEAR_S = 600 s, a CHOICE. The data (Octopus relay logs, v0b-v0f
# V0-xhigh-1, relay*.jsonl: t_end of a conversation's request to t0 of its
# next): 482 gaps with no compaction between them, median 0.48 s, p90 1.73,
# p99 17.1, max 38.2 s (a `terminal` step); every gap over 60 s (11, 81 s
# to 7,984 s) spans the conversation's own compaction, which is not
# idleness. One harness, one task family, benchmark-driven: no human think
# time and no cold install in the sample, so the data is thin and the floor
# the operator set for thin data applies -- 10 minutes, ~16x the longest
# tool step seen. YAMADORI_IDLE_CLEAR_S overrides it. The switch:
# X-Yamadori-Features {"idle_clear": false} or YAMADORI_IDLE_CLEAR=0
# (tiers.BEHAVIOURS; default on so the live measurement can confirm the
# boost).
# ---------------------------------------------------------------------------
IDLE_CLEAR_S = float(os.environ.get("YAMADORI_IDLE_CLEAR_S", "600"))
# proxy sets it: key -> why that conversation must not be cleared now (a
# warm pending), or None.
idle_guard = None


def _idle_default() -> bool:
    v = os.environ.get("YAMADORI_IDLE_CLEAR")
    return (v if v is not None else "1").strip().lower() not in (
        "0", "", "off", "false", "no")


def clear_idle(grant: dict | None, *, on: bool | None = None,
               log: list | None = None, key: str | None = None,
               model: str | None = None, after_s: float | None = None,
               account: str | None = None) -> list[dict]:
    """A request is about to generate on `grant`'s slot: clear every OTHER
    conversation's pinned slot idle longer than IDLE_CLEAR_S. Returns the
    records (also appended to `log`, else the current request's). Never
    raises; a no-op outside the proxy process.

    `after_s` (the live test's X-Yamadori-Features `idle_clear_s`, honoured
    by the proxy for a TEST account only) replaces IDLE_CLEAR_S for this
    request and narrows the candidates to that `account`'s own
    conversations: a test never clears anyone else's slot early."""
    if not grant or grant.get("warm") or grant.get("_server") or not ENABLED or not _release_enabled:
        return []
    ctx = _request_ctx()
    if on is None:
        on = ctx["idle_on"] if ctx and ctx.get("idle_on") is not None \
            else _idle_default()
    if not on:
        return []
    if log is None and ctx:
        log = ctx.get("idle_log")
    mine = key or (ctx or {}).get("key")
    if after_s is None and ctx:
        after_s = ctx.get("idle_after_s")
    if account is None and ctx:
        account = ctx.get("account")
    limit = IDLE_CLEAR_S if after_s is None else max(float(after_s), 0.0)
    now = time.time()
    todo = []
    with _lock:
        if _compacting > 0:
            return []
        for k, s in list(_pins.items()):
            if k == HELPER or k == mine or s == grant.get("slot"):
                continue
            if after_s is not None and (not account
                                        or _key_account.get(k) != account):
                continue
            if s in _cleared or s in _releasing or _busy.get(s, 0) > 0:
                continue
            idle = now - max(_last_end.get(s, _T0), _last_seen.get(k, 0.0))
            if idle <= limit:
                continue
            todo.append((k, s, idle))
    out = []
    for k, s, idle in todo:
        why = None
        if idle_guard is not None:
            try:
                why = idle_guard(k)
            except Exception:                                    # noqa: BLE001
                why = None
        if why:
            continue
        with _lock:
            # Re-checked under the lock: the conversation may have come back.
            if (_pins.get(k) != s or s in _releasing or _busy.get(s, 0) > 0
                    or _compacting > 0 or s in _cleared):
                continue
            ev = threading.Event()
            _releasing[s] = ev
            _busy[s] = _busy.get(s, 0) + 1
            _prompts.pop(s, None)
        res = _send_release(s, ev, model)
        rec = {"slot": s, "why": f"idle conversation ({round(idle)} s > "
                                 f"{round(limit)} s"
                                 + (", test override)" if after_s is not None
                                    else ")"),
               "key": k[:8], "idle_s": round(idle, 1)}
        _fill(rec, res)
        if rec["released"] or res.get("skipped") == "already empty":
            with _lock:
                _cleared[s] = {"cleared_at": round(time.time(), 1),
                               "idle_s": round(idle, 1),
                               "cells_cleared": res.get("cells_before")}
        _save()
        _note(rec, log)
        out.append(rec)
    return out


def _known_n() -> int:
    """The slot count without asking the server (count() may)."""
    return _n if _n is not None else FALLBACK_SLOTS


def _release_default() -> bool:
    v = os.environ.get("YAMADORI_SLOT_RELEASE")
    return (v if v is not None else "1").strip().lower() not in (
        "0", "", "off", "false", "no")


def _helper_run_active() -> bool:
    try:
        import admission
        return admission.helper_active() > 0
    except Exception:                                            # noqa: BLE001
        return False


def _note(rec: dict, log: list | None) -> None:
    if isinstance(log, list):
        log.append(rec)
    with _lock:
        _released_recent.append(dict(rec, at=round(time.time(), 1)))
        del _released_recent[:-32]
    try:
        # The dashboard's history (mcp/stats_store.py; the JJAVA page's
        # lane releases). A queue put; never raises.
        import stats_store
        stats_store.release(rec)
    except Exception:                                            # noqa: BLE001
        pass
    if rec.get("released"):
        print(f"  slot release: slot {rec['slot']} ({rec['why']}) "
              f"cells_before={rec.get('cells_before')} "
              f"method={rec.get('method')} {rec.get('ms')} ms", flush=True)
    else:
        print(f"  slot release: slot {rec['slot']} ({rec['why']}) kept -- "
              f"{rec.get('skipped') or rec.get('error')}", flush=True)


def lane_kept_note(slot: int | None, why: str, log: list | None = None) -> dict | None:
    """THE LANE is kept (lane_kept()): the record a release would have
    written, without the release and without a log line. Never raises."""
    if slot is None:
        return None
    rec = {"slot": slot, "why": why, "released": False,
           "skipped": "the lane is kept (layout v2: slots THE LANE)"}
    if log is None:
        ctx = _request_ctx()
        log = (ctx or {}).get("log")
    if isinstance(log, list):
        log.append(rec)
    try:
        import stats_store                  # the JJAVA page's lane record
        stats_store.release(rec)
    except Exception:                                            # noqa: BLE001
        pass
    return rec


def after_helper_run(what: str) -> dict | None:
    """The helper lane is being let go (admission.helper_lane): release the
    second brain's slot. Called with the lane's inflight count already
    decremented and its semaphore still held, so the next run waits."""
    return release_idle(helper_slot(_known_n()),
                        f"second brain: {what} ended")


def holds() -> dict:
    """Who owns each card's one conversation, for the dashboard (read only, nothing decides from it): the main
    card's owner (the conversation on slot 0 while it has a request in flight or was active less than
    PRIMARY_HOLD_S ago) and the other card's, each {owner (first 8 characters), busy, idle_s, hold_s, rest_s}, None
    when nobody holds it. Only meaningful inside the proxy, where the pins live. `one_conversation` says whether
    the rule is in force (the served /props says 1 or 2 slots; None while the count has not been read: a view
    never makes that read, count() would ask a server)."""
    now = time.time()
    with _lock:
        out: dict = {"hold_s": PRIMARY_HOLD_S, "one_conversation": one_conversation(_n) if _n else None,
                     "main": None, "other": None,
                     "other_model": _other.get("model")}
        for name, (k, busy, since) in (("main", _owner_of_card(now)), ("other", _other_owner(now))):
            if k:
                out[name] = {"owner": k[:8], "busy": bool(busy), "idle_s": round(since, 1),
                             "hold_s": PRIMARY_HOLD_S, "rest_s": _rest_of_hold(busy, since)}
    return out


def occupants() -> dict:
    """What each slot holds, for the dashboard (read only, nothing decides from it): the conversation pinned to
    each slot of the MAIN card (`main`: slot -> {conv, busy, idle_s, held, rest_s}) and the other card's owner on
    its conversation slot (`other`: {slot, conv, busy, idle_s, held, rest_s}). `conv` is the first 8 characters of
    the conversation's key, `idle_s` the seconds since its latest activity (None when this process has not seen
    one: a pin restored at a restart), `held` whether the one-conversation hold still keeps the card for it
    (slot 0 only; None elsewhere) and `rest_s` what is left of that hold (0 when it has ended, None while a
    request is in flight). Unlike holds(), an owner past its hold is still listed: its cells are still there.
    Only meaningful inside the proxy, where the pins live."""
    now = time.time()
    out: dict = {"hold_s": PRIMARY_HOLD_S, "main": {}, "other": None, "other_model": _other.get("model")}
    with _lock:
        for k, s in _pins.items():
            if k == HELPER:
                continue
            ended = _last_end.get(s, 0.0)
            if _restore_stamp.get(s) == ended:
                ended = 0.0
            last = max(ended, _used.get(k, 0.0), _last_seen.get(k, 0.0))
            busy = _busy.get(s, 0) > _warming.get(s, 0)
            idle = round(now - last, 1) if last else None
            row = {"conv": k[:8], "busy": bool(busy), "idle_s": idle, "held": None, "rest_s": None}
            if s == 0:
                row["held"] = bool(busy or (idle is not None and idle < PRIMARY_HOLD_S))
                row["rest_s"] = (None if busy else
                                 max(0, int(PRIMARY_HOLD_S - idle + 0.999)) if idle is not None else 0)
            out["main"][s] = row
        k = _other.get("owner")
        if k:
            busy = _other["busy"] > 0
            last = max(_other["last_end"], _last_seen.get(k, 0.0))
            idle = round(now - last, 1) if last else None
            out["other"] = {"slot": OTHER_CONV_SLOT, "conv": k[:8], "busy": bool(busy), "idle_s": idle,
                            "held": bool(busy or (idle is not None and idle < PRIMARY_HOLD_S)),
                            "rest_s": (None if busy else
                                       max(0, int(PRIMARY_HOLD_S - idle + 0.999)) if idle is not None else 0)}
    return out


def snapshot() -> dict:
    with _lock:
        return {"slots": _n, "source": _n_source,
                "pins": {k[:8]: s for k, s in _pins.items()},
                "helper": helper_slot(_n) if _n else None,
                "child": child_slot(_n) if _n else None,
                "lane": {"slot": child_slot(_n) if _n else None,
                         "ranked": lane_ranked(), "rank": RANK_LANE,
                         "kept": lane_kept()},
                "ranks": {"on": KV_RANK, "hold_s": PRIMARY_HOLD_S,
                          "primary": (_primary or "")[:8] or None,
                          "live": sorted(k[:8] for k in _act
                                         if _live(k, time.time()))},
                "busy": {s: c for s, c in _busy.items() if c},
                "known_prompts": {s: fp["chars"][-1] if fp.get("chars") else 0
                                  for s, fp in _prompts.items()},
                "release": {"enabled": _release_enabled,
                            "default_on": _release_default(),
                            "releasing": sorted(_releasing),
                            "recent": list(_released_recent[-8:])},
                "idle_clear": {"after_s": IDLE_CLEAR_S,
                               "default_on": _idle_default(),
                               "cleared": {s: dict(c) for s, c in
                                           _cleared.items()},
                               "compactions_in_flight": _compacting}}


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
    # The model server's own time for this generation (prompt + predicted,
    # llama-server `timings`): the part of a request's wall clock that is
    # the model's, so the rest is ours (x_yamadori.cache.model_ms).
    ms = None
    if t.get("prompt_ms") is not None and t.get("predicted_ms") is not None:
        ms = round(float(t["prompt_ms"]) + float(t["predicted_ms"]))
    rec = {"prompt": prompt, "reused": reused, "processed": processed,
           "slot": g.get("slot"), "mode": g.get("mode"),
           "evicted": g.get("evicted"), "model_ms": ms}
    # The prefill's own time and the decode rate (llama-server `timings`
    # prompt_ms, predicted_n / predicted_ms): what a cold resume costs and
    # what an idle slot takes from a decode (IDLE CLEAR, the live `slots`
    # test). None when the server did not report them.
    rec["prompt_ms"] = (round(float(t["prompt_ms"]))
                        if t.get("prompt_ms") is not None else None)
    rec["decode_tps"] = (round(float(t["predicted_n"]) * 1000.0
                               / float(t["predicted_ms"]), 2)
                         if t.get("predicted_n") and t.get("predicted_ms")
                         else None)
    if g.get("affinity"):
        rec["affinity"] = g["affinity"]
    if g.get("adopted"):
        rec["adopted"] = g["adopted"]
    if g.get("mode") == "compaction":
        # LAYOUT V2: where a compaction sent as is went, and whose cache it
        # displaced (x_yamadori.cache).
        rec["how"] = g.get("how")
        if g.get("displaced"):
            rec["displaced"] = g["displaced"]
    if g.get("stray_pins_dropped"):
        rec["stray_pins_dropped"] = g["stray_pins_dropped"]
    if g.get("resumed_cold"):
        rec["resumed_cold"] = resumed_how(g["resumed_cold"], rec)
    if g.get("kv_ranks") is not None:
        # RANKS: what the engine was told for this generation.
        rec["kv_rank"] = g.get("kv_rank")
        rec["kv_ranks"] = list(g["kv_ranks"])
        rec["primary"] = g.get("primary")
    return rec


# A slot IDLE CLEAR emptied holds only the release's one-token prompt, so a
# resumed request that REUSES more than this many tokens got them from
# llama-server's host-RAM prompt cache (--cache-ram, default 8192 MiB; not
# set in config.yaml): the release's prompt diverges from the slot's at
# position 0, so get_available_slot (server-context.cpp, f_keep < 0.5) SAVED
# the slot's state to host RAM before clearing it, and the resumed request
# LOADS it back (server_prompt_cache::load, the most similar entry keeping
# >= 25% of it). Live 2026-09-27: a ~41k-token conversation came back in
# 516 ms (the live `slots` test) and 1,141-1,214 ms (the placement probe,
# n=3), not the ~30 s a re-prefill costs. An entry evicted past the cache's
# limit, or a state bigger than it, is re-processed instead.
RESTORED_MIN_TOKENS = 64


def resumed_how(cold: dict, rec: dict) -> dict:
    """x_yamadori.slots.resumed_cold for a conversation's first generation
    after IDLE CLEAR: the clear ({slot, cleared_at, idle_s, cells_cleared})
    plus how the prompt came back -- `restored` (from the server's host-RAM
    prompt cache) or `reprocessed` (prefilled again) -- with the tokens
    reused and processed and the prefill's ms. `how` is None when the
    server reported no counts."""
    out = dict(cold)
    reused, processed = rec.get("reused"), rec.get("processed")
    if reused is None or processed is None:
        how = None
    elif int(reused) >= max(RESTORED_MIN_TOKENS,
                            0.5 * (int(reused) + int(processed))):
        how = "restored"
    else:
        how = "reprocessed"
    out.update(how=how, reused=reused, processed=processed,
               prompt_ms=rec.get("prompt_ms"))
    return out


if __name__ == "__main__":
    print(json.dumps({"slots": count(), "source": source()}, indent=1))
