#!/usr/bin/env python
"""Slot placement (mcp/slots.py). No GPU, no server.

    python mcp/test_slots.py      -> "N/M checks passed"

WHAT THIS IS GATING (#10 and #11 in docs/SELF-IMPROVEMENT-LOG.md): the second
brain never runs on a conversation's slot.

Live gate, 2026-09-24 12:49, `test_live_stack` cache test: the conversation
landed on slot 2 by evicting another (`evicted: ba1ade26`), where at 12:15 the
same test sat on slot 0; then its warms re-read the whole prompt (reused 0 of
4567 and of 14399). Two ways the old placement put the second brain on a
conversation's slot, both read from the code:

  1. IN THIS PROCESS. Deep thinking runs BEFORE the conversation's own turn
     acquires its slot, so the conversation's last use is its previous turn.
     With every pinnable slot held, the second brain (an LRU pin like any
     other) evicted the least recently used pin -- which could be the very
     conversation it was working for -- and ran its hops there.
  2. ACROSS PROCESSES. The worker and the tools API keep their own copy of the
     table, where HELPER took "the highest pinnable slot", slot 2 -- while in
     the proxy a conversation took slot 2 whenever 0 and 1 were held.

The rule now: the CHILD slot, child_slot() = helper_slot() = n-1 (slot 2 of
3, operator 2026-09-28), is the second brain's in every process -- and the
side calls', the decider's and an as-sent compaction's, queued there -- and no
conversation is ever pinned to it. And every grant carries the slots' KV
RANKS (the primary conversation above the other, the child at the rank of
the conversation it works for) for engine patch 0041.

REMOVED 2026-09-29 (docs/REMOVED.md; the way back is commit e360d37): the
second brain (shomen: deep thinking, fan-out's B/C, fix-ups) and with it the
fan-out release check (one lane held across B and C). The placement and
release rules it drove stay -- slots.HELPER and admission.helper_lane serve
whatever helper work remains (model.post's helper role, summaries) -- so the
checks below still say "second brain" / "deep thinking" for the multi-hop
helper job they simulate.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["YAMADORI_SLOTS"] = "3"

import slots  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def _hold(key: str) -> dict:
    """A conversation's turn: acquire and release (its pin stays)."""
    g = slots.acquire(key)
    slots.release(g)
    return g


def test_the_helper_slot_is_fixed_and_reserved():
    slots.reset(n=3)
    check(slots.helper_slot(3) == 2 and slots.child_slot(3) == 2
          and slots._transient_slot(3) == 2 and slots.helper_slot(4) == 3
          and slots.helper_slot(2) == 1 and slots.helper_slot(1) is None,
          "helper_slot = child_slot = the transient slot = n-1 (slot 2 of 3), "
          "the same number in every process; none with one slot")
    check(slots.conversation_slots(3) == [0, 1]
          and slots.conversation_slots(4) == [0, 1, 2]
          and slots.FALLBACK_SLOTS == 3,
          "conversations live on slots 0 .. n-2; the fallback count is "
          "config.yaml's -np 3, nothing assumes 4",
          str(slots.conversation_slots(3)))
    got = [_hold(f"conv-{i}")["slot"] for i in range(5)]
    check(2 not in got,
          "five conversations in turn are never pinned to the child slot (2)",
          str(got))
    check(set(got) == {0, 1}, "they share slots 0 and 1 by LRU", str(got))
    g = slots.acquire(slots.HELPER)
    slots.release(g)
    check(g["slot"] == 2 and g["mode"] == "helper" and not g["evicted"],
          "the second brain gets the child slot (2) and evicts nobody",
          str(g))
    check(slots.HELPER not in slots._pins,
          "the second brain holds no pin (nothing can evict it, it can "
          "evict nothing)", str(slots._pins))


def test_the_second_brain_never_takes_the_conversation_it_serves():
    """The 12:49 shape: every conversation slot pinned, and the conversation
    being served is the least recently used (its last turn was long ago);
    deep thinking runs before its turn acquires anything."""
    slots.reset(n=3)
    _hold("served")                 # the conversation's earlier turn
    _hold("other-a")                # every conversation slot now pinned
    served_slot = slots._pins.get("served")
    before = dict(slots._pins)
    hops = []
    for _ in range(13):             # deep thinking: 13 hops
        g = slots.acquire(slots.HELPER)
        hops.append(g["slot"])
        slots.release(g)
    check(all(s == 2 for s in hops),
          "all 13 deep-thinking hops run on the child slot", str(hops))
    check(slots._pins == before,
          "no conversation lost its pin to the second brain",
          f"{before} -> {slots._pins}")
    g = slots.acquire("served")
    slots.release(g)
    check(g["slot"] == served_slot and not g["evicted"],
          "the served conversation's turn then finds its own slot "
          "(its cached prefix intact)", str(g))


def test_a_busy_helper_slot_defers_rather_than_moving():
    slots.reset(n=3)
    _hold("c0")
    _hold("c1")
    a = slots.acquire(slots.HELPER)
    b = slots.acquire(slots.HELPER)       # e.g. a summary during fan-out
    check(a["slot"] == b["slot"] == 2 and "queued" in b["how"],
          "a second concurrent second-brain call still targets slot 2 (the "
          "server defers it) instead of moving onto a conversation's slot",
          f"{a} {b}")
    slots.release(a)
    slots.release(b)


def test_transient_calls_spare_conversations():
    slots.reset(n=3)
    _hold("c0")
    _hold("c1")
    t = slots.acquire(None, transient=True)
    check(t["slot"] == 2 and t["mode"] == "transient",
          "a side call takes the child slot", str(t))
    t2 = slots.acquire(None, transient=True)
    check(t2["slot"] == 2 and not t2["evicted"] and "queued" in t2["how"],
          "with the child slot busy, the next QUEUES there (the server "
          "defers it) -- it never evicts a conversation", str(t2))
    h = slots.acquire(slots.HELPER)
    check(h["slot"] == 2 and "queued" in h["how"],
          "and the second brain queues on the same slot", str(h))
    slots.release(h)
    slots.release(t)
    slots.release(t2)
    check(slots._pins.get("c0") is not None
          and slots._pins.get("c1") is not None,
          "both conversations keep their pins", str(slots._pins))


def test_a_warm_goes_to_the_conversations_own_slot():
    slots.reset(n=3)
    g = _hold("conv")
    w = slots.acquire("conv", warm=True)
    slots.release(w)
    check(w["slot"] == g["slot"] and w["mode"] == "warm",
          "a warm targets the slot the conversation generated on",
          f"{g} {w}")


# ADOPTION (#38, operator 2026-09-25: "why does it jump around?"). A
# conversation whose KEY changes -- a compaction continuation the link
# missed, the no-system-message gap, a restart -- had no pin and took a free
# slot or evicted another conversation's, abandoning the slot that held its
# own prefix (live: "slot 1 (pinned) first prompt 14227 reused 0 ...
# evicted 7b43d48b" while slot 0 held the conversation).
TOOLS = [{"type": "function", "function": {
    "name": "terminal", "description": "run a command " * 200,
    "parameters": {"type": "object", "properties": {}}}}]
SYSTEM = {"role": "system", "content": "You are Hermes Agent. " * 300}


def _fp(*msgs: dict) -> dict:
    return slots.fingerprint({"tools": TOOLS, "messages": [SYSTEM, *msgs]})


def _turn(key: str, fp: dict, **kw) -> dict:
    """A conversation's turn that finished with an answer: the slot now
    holds `fp`."""
    g = slots.acquire(key, prompt=fp, **kw)
    slots.remember(g, fp)
    slots.release(g)
    return g


USER = {"role": "user", "content": "build a space shooter " * 100}
CALL = {"role": "assistant", "content": "", "tool_calls": [
    {"id": "c1", "type": "function",
     "function": {"name": "terminal", "arguments": "{\"command\": \"ls\"}"}}]}
RESULT = {"role": "tool", "tool_call_id": "c1", "content": "js/game.js"}


def test_a_new_key_adopts_the_slot_its_prompt_continues():
    slots.reset(n=3)
    a = _turn("conv-a", _fp(USER, CALL, RESULT))
    other = _turn("conv-other", _fp({"role": "user", "content": "plan a "
                                     "garden " * 100}))
    check(a["slot"] == 0 and other["slot"] == 1, "two conversations, 0 and 1",
          f"{a['slot']} {other['slot']}")
    # The same conversation under a NEW key, one turn further on.
    later = _fp(USER, CALL, RESULT, {"role": "assistant", "content": "done"},
                {"role": "user", "content": "now add a score counter"})
    g = slots.acquire("conv-a-renamed", prompt=later)
    slots.release(g)
    check(g["slot"] == 0 and not g["evicted"] and g.get("adopted")
          and g["adopted"]["key"] == "conv-a"[:8]
          and g["how"].startswith("adopted"),
          "a key with no pin whose prompt continues slot 0's (through an "
          "answer) adopts slot 0 -- no eviction, no abandoned slot",
          str(g))
    check(slots._pins.get("conv-a-renamed") == 0 and "conv-a" not in slots._pins,
          "and the pin moves to the new key", str(slots._pins))
    old = slots.acquire("conv-b", prompt=later)
    slots.release(old)
    check(old["slot"] != 0 and not old.get("adopted"),
          "a slot granted again (its claim dropped: its content is about to "
          "change) is not adopted by yet another key", str(old))


def test_a_shared_harness_head_is_not_enough():
    """Two conversations of one harness share tools + system prompt (here
    ~2,600 tokens): that must never move one onto the other's slot."""
    slots.reset(n=3)
    _turn("conv-a", _fp(USER, CALL, RESULT))
    g = slots.acquire("conv-b", prompt=_fp(
        {"role": "user", "content": "plan a garden " * 100}))
    slots.release(g)
    check(g["slot"] == 1 and not g.get("adopted"),
          "a different conversation with the same head takes a free slot",
          str(g))
    slots.reset(n=3)
    _turn("row-1", _fp(USER))
    g = slots.acquire("row-2", prompt=_fp(USER))
    slots.release(g)
    check(g["slot"] == 1 and not g.get("adopted"),
          "an identical opening (benchmark rows, two session tokens) is not "
          "a continuation: nothing generated is shared", str(g))
    slots.reset(n=3)
    _turn("gap-1", _fp(USER))
    g = slots.acquire("gap-2", prompt=_fp(USER, CALL, RESULT))
    slots.release(g)
    check(g["slot"] == 0 and g.get("adopted", {}).get("why")
          == "extends its whole prompt",
          "a prompt that strictly extends a slot's whole prompt adopts it "
          "(nebari.key_of's no-system-message gap: turn 2's key differs)",
          str(g))


def test_a_busy_or_short_slot_is_not_adopted():
    slots.reset(n=3)
    fp = _fp(USER, CALL, RESULT)
    _turn("conv-a", fp)
    held = slots.acquire("conv-a")             # busy: an orphan still runs
    g = slots.acquire("conv-new", prompt=_fp(USER, CALL, RESULT, {
        "role": "assistant", "content": "x"}))
    slots.release(g)
    slots.release(held)
    check(g["slot"] != 0 and not g.get("adopted"),
          "a busy slot is never adopted", str(g))
    slots.reset(n=3)
    small = slots.fingerprint({"messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant",
                                            "content": "hello"}]})
    _turn("tiny", small)
    g = slots.acquire("tiny-2", prompt=slots.fingerprint({"messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant",
                                            "content": "hello"},
        {"role": "user", "content": "more"}]}))
    slots.release(g)
    check(not g.get("adopted"),
          f"under AFFINITY_MIN_TOKENS ({slots.AFFINITY_MIN_TOKENS}) shared, "
          f"nothing is adopted", str(g))


def test_pins_survive_a_restart():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(prefix="yamadori_slots_"),
                        "slots_state.json")
    slots.reset(n=3)
    slots.persist(path)
    try:
        _turn("conv-x", _fp(USER))
        _turn("conv-y", _fp({"role": "user", "content": "plan a garden "
                                                        * 100}))
        before = dict(slots._pins)
        slots.reset(n=3)                        # the restart: memory gone
        st = slots.persist(path)
        check(st["pins"] == 2 and dict(slots._pins) == before
              and st["prompts"] == 2,
              "the pins and what each slot holds come back from the file",
              f"{st} {slots._pins}")
        g = slots.acquire("conv-y")
        slots.release(g)
        check(g["slot"] == before["conv-y"] and g["how"].startswith("pinned"),
              "so a conversation goes back to its own slot after a restart "
              "(it used to re-pin in arrival order)", str(g))
    finally:
        slots._state_path = None
        slots.reset(n=3)



# ---------------------------------------------------------------------------
# RELEASE (slots.py): the second brain's slot after each run, the transient
# slot after each side call -- never a conversation's, never while held.
# model.release_slot is replaced: no request reaches a model server.
# ---------------------------------------------------------------------------
class _Releases:
    """Stands in for model.release_slot and records each call."""

    def __init__(self, gate=None):
        self.calls: list[int] = []
        self.gate = gate

    def __call__(self, slot, model=None, timeout=None):
        if self.gate is not None:
            self.gate.wait(5)
        self.calls.append(slot)
        return {"ok": True, "method": "shrink", "cells_before": 41234,
                "ms": 7}


def _with_release(fn):
    """Run fn(releases) with release enabled in this process and
    model.release_slot replaced; restore everything after."""
    import model
    saved = (model.release_slot, os.environ.get("YAMADORI_SLOT_RELEASE"))
    rel = _Releases()
    model.release_slot = rel
    os.environ.pop("YAMADORI_SLOT_RELEASE", None)
    slots.reset(n=3)
    slots.enable_release(True)
    try:
        fn(rel)
    finally:
        slots.enable_release(False)
        model.release_slot = saved[0]
        if saved[1] is None:
            os.environ.pop("YAMADORI_SLOT_RELEASE", None)
        else:
            os.environ["YAMADORI_SLOT_RELEASE"] = saved[1]
        slots.reset(n=3)


def _hop() -> dict:
    """One second-brain generation (model.post's own acquire/release)."""
    g = slots.acquire(slots.HELPER)
    slots.release(g)
    return g


def test_the_helper_slot_is_released_after_a_run_not_between_hops():
    import admission
    import cancel

    def body(rel):
        log: list = []
        tok = cancel.Token()
        slots.bind_request(tok, True, "default", log)
        with cancel.bound(tok):
            with admission.helper_lane(what="investigate") as got:
                check(got, "the lane is granted")
                _hop()
                _hop()
                check(rel.calls == [],
                      "two hops of one job: nothing released between them "
                      "(they reuse each other's prefix)", str(rel.calls))
        check(rel.calls == [2],
              "the run ended: the helper slot (2) released once",
              str(rel.calls))
        check(len(log) == 1 and log[0]["slot"] == 2 and log[0]["released"]
              and log[0]["cells_before"] == 41234 and log[0]["ms"] == 7
              and log[0]["method"] == "shrink"
              and "investigate" in log[0]["why"],
              "and the request's record says which slot, how many cells, "
              "how long, how (x_yamadori.slots.released)", str(log))
        check(slots.snapshot()["busy"] == {} and 2 not in slots._dirty,
              "the slot is idle and clean afterwards", str(slots.snapshot()))
        with admission.helper_lane(what="investigate"):
            pass
        check(rel.calls == [2],
              "a run that generated nothing releases nothing (the slot is "
              "not dirty: no request to the server)", str(rel.calls))
    _with_release(body)


def test_never_a_pinned_slot():
    def body(rel):
        # Two side calls at once: both on the child slot (2), never on the
        # unpinned conversation slot 1...
        _hold("conv-a")
        busy2 = slots.acquire(None, transient=True)
        side = slots.acquire(None, transient=True)
        slots.release(side)
        slots.release(busy2)
        check(busy2["slot"] == 2 and side["slot"] == 2,
              "side calls queue on the child slot, even with slot 1 free",
              str([busy2, side]))
        # ...and a conversation slot marked as filled by this process (the
        # 4-slot layout's side call on a free slot) is never released once a
        # conversation pins it.
        conv = slots.acquire("conv-b")
        slots.release(conv)
        slots._dirty.add(1)
        check(conv["slot"] == 1, "a conversation now holds slot 1", str(conv))
        r = slots.release_idle(1, "side call ended")
        check(rel.calls == [] and r and r["released"] is False
              and "pinned" in r["skipped"],
              "a pinned slot is never released, even one a side call "
              "filled", str(r))
        r0 = slots.release_idle(0, "anything")
        check(r0 is None and rel.calls == [],
              "a conversation's slot this process never filled as helper or "
              "transient: nothing attempted", str(r0))
        # Compaction affinity: placed ON a conversation's slot; mode
        # "affinity" never marks it for release.
        slots._busy[0] = 1
        slots.release({"slot": 0, "mode": "affinity"})
        check(0 not in slots._dirty,
              "a compaction served on a conversation's slot by affinity "
              "leaves it the conversation's", str(slots._dirty))
        r2 = slots.release_idle(2, "side call ended")
        check(r2 and r2["released"] and rel.calls == [2],
              "the child slot a side call filled: released", str(r2))
    _with_release(body)


def test_never_while_held():
    import admission

    def body(rel):
        g = slots.acquire(slots.HELPER)
        slots._dirty.add(2)
        r = slots.after_helper_run("investigate")
        check(rel.calls == [] and r and "this process" in r["skipped"],
              "a request in flight on the slot: not released", str(r))
        slots.release(g)
        with admission.helper_lane(what="investigate"):
            r = slots.release_idle(2, "side call ended")
            check(rel.calls == [] and r
                  and "second-brain run" in r["skipped"],
                  "a second-brain run holding the lane: the helper slot "
                  "is not released by anyone else (a transient call that "
                  "borrowed it)", str(r))
        check(rel.calls == [2], "and is released when that run ends",
              str(rel.calls))
    _with_release(body)


def test_a_new_request_waits_for_a_release_in_flight():
    import threading
    import time
    import model

    def body(rel):
        gate = threading.Event()
        slow = _Releases(gate)
        model.release_slot = slow
        _hop()
        t = threading.Thread(target=slots.after_helper_run,
                             args=("investigate",))
        t.start()
        for _ in range(200):
            if 2 in slots._releasing:
                break
            time.sleep(0.005)
        order: list = []

        def nxt():
            g = slots.acquire(slots.HELPER)
            order.append(("granted", list(slow.calls),
                          g.get("waited_release_ms")))
            slots.release(g)
        t2 = threading.Thread(target=nxt)
        t2.start()
        time.sleep(0.05)
        check(order == [], "the next run's first hop waits while the "
              "release is on its way", str(order))
        gate.set()
        t.join(5)
        t2.join(5)
        check(order and order[0][1] == [2] and order[0][2] is not None,
              "and is sent only after it: never overwritten by it",
              str(order))
    _with_release(body)


def test_the_switch():
    import tiers
    import cancel

    def body(rel):
        os.environ["YAMADORI_SLOT_RELEASE"] = "0"
        _hop()
        check(slots.after_helper_run("investigate") is None
              and rel.calls == [],
              "YAMADORI_SLOT_RELEASE=0: nothing released (the old "
              "behaviour)", str(rel.calls))
        os.environ.pop("YAMADORI_SLOT_RELEASE", None)
        tok = cancel.Token()
        slots.bind_request(tok, False, "header", [])
        with cancel.bound(tok):
            check(slots.after_helper_run("investigate") is None
                  and rel.calls == [],
                  "a request whose header switched it off: nothing "
                  "released", str(rel.calls))
        check(slots.after_helper_run("investigate") and rel.calls == [2],
              "switched on (the default): released", str(rel.calls))
        slots.enable_release(False)
        _hop()
        check(slots.after_helper_run("investigate") is None
              and rel.calls == [2],
              "a process that never enabled release (the worker, the "
              "tools API, a test suite) sends nothing", str(rel.calls))
    _with_release(body)
    t = tiers.resolve({}, tiers.from_header('{"slot_release": false}'))
    check(tiers.behaviour_source(t, "slot_release") == (False, "header"),
          "X-Yamadori-Features {\"slot_release\": false} forces it off",
          str(tiers.behaviour_source(t, "slot_release")))
    old = os.environ.pop("YAMADORI_SLOT_RELEASE", None)
    try:
        check(tiers.behaviour_source(tiers.resolve({}), "slot_release")
              == (True, "default"), "on by default")
    finally:
        if old is not None:
            os.environ["YAMADORI_SLOT_RELEASE"] = old


def test_model_release_slot():
    """The one door's release: /slots first, erase when the server allows
    it, else the shrink; a slot another process is generating on is left
    alone. urlopen is replaced: no server."""
    import io
    import json
    import urllib.error
    import urllib.request
    import model
    saved = (urllib.request.urlopen, model._erase_ok)
    sent: list = []
    state = {"table": [{"id": 2, "is_processing": False,
                        "n_prompt_tokens": 38000}], "erase": 501}

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/running"):
            # llama-swap lists bonsai ready: the release may read its /slots
            # (model.release_slot asks /running first since 2026-09-30)
            return _R(json.dumps({"running": [{"model": "bonsai", "state": "ready"}]}).encode())
        body = None if isinstance(req, str) else json.loads(req.data)
        sent.append((url.split("/upstream/bonsai")[-1], body))
        if url.endswith("/slots"):
            return _R(json.dumps(state["table"]).encode())
        if "action=erase" in url:
            if state["erase"] != 200:
                raise urllib.error.HTTPError(url, state["erase"], "no",
                                             {}, None)
            return _R(b'{"id_slot": 2, "n_erased": 38000}')
        return _R(b'{"content": "", "timings": {"prompt_n": 1}}')
    urllib.request.urlopen = fake
    model._erase_ok = None
    try:
        r = model.release_slot(2)
        check(r["ok"] and r["method"] == "shrink"
              and r["cells_before"] == 38000
              and r.get("erase_refused") == "HTTP 501"
              and [u for u, _ in sent] == ["/slots", "/slots/2?action=erase",
                                           "/completion"]
              and sent[-1][1] == {"prompt": model.SHRINK_PROMPT,
                                  "n_predict": 0, "id_slot": 2,
                                  "cache_prompt": True},
              "no --slot-save-path (501): the shrink, one token of its own "
              "on that slot", json.dumps([r, sent]))
        sent.clear()
        r = model.release_slot(2)
        check([u for u, _ in sent] == ["/slots", "/completion"],
              "the refusal is remembered: erase is not asked again",
              str(sent))
        model._erase_ok, state["erase"] = None, 200
        sent.clear()
        r = model.release_slot(2)
        check(r["method"] == "erase" and [u for u, _ in sent]
              == ["/slots", "/slots/2?action=erase"],
              "a server that allows erase: erase, no shrink", str(sent))
        state["table"] = [{"id": 2, "is_processing": True,
                           "n_prompt_tokens": 500}]
        sent.clear()
        r = model.release_slot(2)
        check(not r["ok"] and "processing" in r["skipped"]
              and [u for u, _ in sent] == ["/slots"],
              "another process generating on it: left alone, nothing sent",
              json.dumps([r, sent]))
        state["table"] = [{"id": 2, "is_processing": False}]
        sent.clear()
        r = model.release_slot(2)
        check(r["ok"] and r.get("skipped") == "already empty"
              and len(sent) == 1, "an empty slot: nothing to send", str(r))

        def slots_down(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            if url.endswith("/running"):
                return _R(json.dumps({"running": [{"model": "bonsai", "state": "ready"}]}).encode())
            raise OSError("connection refused")
        urllib.request.urlopen = slots_down
        r = model.release_slot(2)
        check(not r["ok"] and r.get("error", "").startswith("/slots"),
              "the server unreachable: an error recorded, never raised",
              str(r))

        # A READ NEVER LOADS A MODEL (2026-09-30): llama-swap unreadable, or
        # the model not in its /running -- nothing is asked of /upstream.
        sent.clear()

        def down(req, timeout=None):
            raise OSError("connection refused")
        urllib.request.urlopen = down
        r = model.release_slot(2)
        check(not r["ok"] and "could not be read" in str(r.get("skipped")),
              "llama-swap /running unreadable: skipped with why, nothing asked",
              str(r))

        def not_loaded(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            if url.endswith("/running"):
                return _R(json.dumps({"running": [{"model": "embeddings", "state": "ready"}]}).encode())
            sent.append((url, None))
            raise AssertionError("asked /upstream for a model that is not loaded")
        urllib.request.urlopen = not_loaded
        r = model.release_slot(2)
        check(not r["ok"] and "not loaded" in str(r.get("skipped")) and not sent,
              "the model not loaded: nothing to release, /upstream never asked",
              json.dumps([r, sent]))
    finally:
        urllib.request.urlopen, model._erase_ok = saved



# ---------------------------------------------------------------------------
# IDLE CLEAR (slots.py): another conversation's pinned slot, idle past
# IDLE_CLEAR_S, is cleared when a request is about to generate -- pin kept,
# its next request marked resumed_cold. model.release_slot is replaced.
# ---------------------------------------------------------------------------
def _age(key: str, seconds: float) -> None:
    """Make `key`'s slot look idle for `seconds`."""
    import time
    s = slots._pins[key]
    slots._last_end[s] = time.time() - seconds
    slots._last_seen[key] = time.time() - seconds


def _gen(key, **kw) -> tuple[dict, list]:
    """A generation for `key` (a conversation, HELPER or None): acquire,
    clear_idle as proxy._post_events does, release."""
    g = slots.acquire(key, transient=key is None)
    recs = slots.clear_idle(g, key=key if key not in (None, slots.HELPER)
                            else None, **kw)
    slots.release(g)
    return g, recs


def test_an_idle_conversation_is_cleared_for_an_active_one():
    def body(rel):
        os.environ.pop("YAMADORI_IDLE_CLEAR", None)
        _hold("away")
        _hold("here")
        away = slots._pins["away"]
        _age("away", slots.IDLE_CLEAR_S + 60)
        log: list = []
        g, recs = _gen("here", log=log)
        check(rel.calls == [away] and len(log) == 1
              and log[0]["slot"] == away and log[0]["released"]
              and log[0]["idle_s"] > slots.IDLE_CLEAR_S
              and log[0]["key"] == "away"[:8],
              "a request about to generate clears the OTHER conversation's "
              "slot idle past IDLE_CLEAR_S (x_yamadori.slots.cleared_idle)",
              str(log))
        check(slots._pins.get("away") == away,
              "the pin is kept: the conversation keeps its slot", str(slots._pins))
        _gen("here")
        check(rel.calls == [away], "a cleared slot is not cleared again",
              str(rel.calls))
        g = slots.acquire("away")
        slots.release(g)
        rc = g.get("resumed_cold") or {}
        check(g["slot"] == away and rc.get("slot") == away
              and rc.get("cells_cleared") == 41234
              and rc.get("idle_s", 0) > slots.IDLE_CLEAR_S,
              "its next request goes back to its own slot and says it "
              "resumed cold (x_yamadori.slots.resumed_cold)", str(g))
        rec = slots.cache_record({"cache_n": 0, "prompt_n": 9000,
                                  "prompt_ms": 8800.4}, None, g)
        got = rec.get("resumed_cold") or {}
        check({k: got.get(k) for k in rc} == rc
              and got.get("how") == "reprocessed" and got.get("reused") == 0
              and got.get("processed") == 9000 and got.get("prompt_ms") == 8800,
              "the cache record carries it beside prompt/reused/processed, "
              "and says the prompt was re-processed (and in how many ms)",
              str(rec))
        rec = slots.cache_record({"cache_n": 41230, "prompt_n": 4,
                                  "prompt_ms": 516.2}, None, g)
        got = rec.get("resumed_cold") or {}
        check(got.get("how") == "restored" and got.get("reused") == 41230
              and got.get("prompt_ms") == 516,
              "a prompt the server brought back from its host-RAM prompt "
              "cache (reused, not processed) is recorded as restored",
              str(rec))
        got = slots.cache_record(None, None, g).get("resumed_cold") or {}
        check("how" in got and got["how"] is None,
              "no counts from the server: how is unknown (None), never a "
              "guess", str(got))
        g2 = slots.acquire("away")
        slots.release(g2)
        check("resumed_cold" not in g2, "once only: the request after that "
              "is warm again", str(g2))
    _with_release(body)


def test_idle_is_measured_and_guarded():
    import time

    def body(rel):
        os.environ.pop("YAMADORI_IDLE_CLEAR", None)
        _hold("away")
        _hold("here")
        away = slots._pins["away"]
        _age("away", slots.IDLE_CLEAR_S - 30)
        _gen("here")
        check(rel.calls == [], "idle for less than IDLE_CLEAR_S: kept",
              str(rel.calls))
        # A tool step longer than every one observed is still far below it.
        check(slots.IDLE_CLEAR_S >= 600, "IDLE_CLEAR_S is 10 minutes "
              "(the relay data's longest tool gap: 38.2 s)",
              str(slots.IDLE_CLEAR_S))
        _age("away", slots.IDLE_CLEAR_S + 60)
        w = slots.acquire("away", warm=True)
        _gen("here")
        check(rel.calls == [], "its warm in flight: kept", str(rel.calls))
        slots.release(w)
        check(time.time() - slots._last_end[away] < 5,
              "and idleness is measured from the END of that warm")
        _gen("here")
        check(rel.calls == [], "so right after its warm it is not idle",
              str(rel.calls))
        _age("away", slots.IDLE_CLEAR_S + 60)
        slots.idle_guard = lambda k: "a warm is pending" if k == "away" \
            else None
        try:
            _gen("here")
            check(rel.calls == [], "a warm pending (held, not yet running): "
                  "kept", str(rel.calls))
        finally:
            slots.idle_guard = None
        busy = slots.acquire("away")
        _age("away", slots.IDLE_CLEAR_S + 60)
        _gen("here")
        check(rel.calls == [], "a request of its own in flight: kept",
              str(rel.calls))
        slots.release(busy)
        _age("away", slots.IDLE_CLEAR_S + 60)
        comp = slots.acquire(None, transient=True,
                             prefix=_fp({"role": "user", "content": "x"}))
        _gen("here")
        check(rel.calls == [], "a compaction sent up as is in flight (it "
              "names no conversation): nothing cleared", str(rel.calls))
        slots.release(comp)
        # LAYOUT V2: the compaction sent as is ran on the least recently used
        # conversation slot -- away's -- and the proxy empties it after
        # (pinned_ok, proxy._post_events); the idle clear then has nothing
        # left to do there.
        check(comp.get("slot") == away and comp.get("displaced") == "away"[:8],
              "the compaction took the least recently used conversation "
              "slot (away's)", str(comp))
        slots.release_idle(comp["slot"], "compaction sent as is ended",
                           pinned_ok=True)
        _gen("here")
        check(rel.calls == [away], "the compaction done: that slot emptied",
              str(rel.calls))
    _with_release(body)


def test_a_lone_conversation_never_clears_itself():
    import cancel

    def body(rel):
        os.environ.pop("YAMADORI_IDLE_CLEAR", None)
        _hold("lone")
        _age("lone", slots.IDLE_CLEAR_S * 5)
        _gen("lone")
        check(rel.calls == [], "the conversation that comes back after an "
              "hour: its own slot is kept", str(rel.calls))
        _age("lone", slots.IDLE_CLEAR_S * 5)
        tok = cancel.Token()
        slots.bind_request(tok, True, "default", [], key="lone",
                           idle_on=True, idle_log=[])
        _age("lone", slots.IDLE_CLEAR_S * 5)
        with cancel.bound(tok):
            _gen(slots.HELPER)
            _gen(None)
        check(rel.calls == [], "nor does deep thinking or a side call "
              "working for it clear it before its own turn", str(rel.calls))
    _with_release(body)


def test_the_idle_switch():
    import tiers
    import cancel

    def body(rel):
        _hold("away")
        _hold("here")
        _age("away", slots.IDLE_CLEAR_S + 60)
        os.environ["YAMADORI_IDLE_CLEAR"] = "0"
        _gen("here")
        check(rel.calls == [], "YAMADORI_IDLE_CLEAR=0: kept (the old "
              "behaviour)", str(rel.calls))
        os.environ.pop("YAMADORI_IDLE_CLEAR", None)
        tok = cancel.Token()
        slots.bind_request(tok, True, "default", [], key="here",
                           idle_on=False, idle_log=[])
        with cancel.bound(tok):
            g = slots.acquire("here")
            slots.clear_idle(g)
            slots.release(g)
        check(rel.calls == [], "a request whose header switched it off: "
              "kept", str(rel.calls))
        slots.enable_release(False)
        _gen("here")
        check(rel.calls == [], "a process that never enabled release: "
              "kept", str(rel.calls))
        slots.enable_release(True)
        _gen("here")
        check(len(rel.calls) == 1, "on (the default): cleared",
              str(rel.calls))
    _with_release(body)
    t = tiers.resolve({}, tiers.from_header('{"idle_clear": false}'))
    check(tiers.behaviour_source(t, "idle_clear") == (False, "header"),
          "X-Yamadori-Features {\"idle_clear\": false} forces it off")
    old = os.environ.pop("YAMADORI_IDLE_CLEAR", None)
    try:
        check(tiers.behaviour_source(tiers.resolve({}), "idle_clear")
              == (True, "default"), "on by default")
    finally:
        if old is not None:
            os.environ["YAMADORI_IDLE_CLEAR"] = old



def test_the_test_override_touches_only_its_own_account():
    """The live `slots` test cannot wait 10 minutes: X-Yamadori-Features
    {"idle_clear_s": N} (honoured by the proxy for a TEST account only)
    replaces the threshold for one request -- and then only that account's
    own idle conversations are candidates."""
    import cancel

    def body(rel):
        os.environ.pop("YAMADORI_IDLE_CLEAR", None)
        tok_other = cancel.Token()
        slots.bind_request(tok_other, True, "default", [], key="other",
                           account="someone-else")
        _hold("other")
        slots.bind_request(cancel.Token(), True, "default", [], key="mine-b",
                           account="live-test")
        _hold("mine-b")
        _age("other", 30)
        _age("mine-b", 30)
        # Every conversation slot is pinned: the active one borrows the
        # transient slot (a side call of the test's own account).
        tok = cancel.Token()
        log: list = []
        slots.bind_request(tok, True, "default", [], key=None,
                           idle_on=True, idle_log=log, account="live-test",
                           idle_after_s=2)
        with cancel.bound(tok):
            _gen(None)
        mine = slots._pins["mine-b"]
        check(rel.calls == [mine] and [r["slot"] for r in log] == [mine]
              and "test override" in log[0]["why"],
              "idle 30 s, threshold 2 s: the test account's own idle "
              "conversation is cleared", str(log))
        check(slots._pins["other"] not in rel.calls,
              "another account's conversation, idle as long, is not "
              "(the override narrows the candidates to its own account)",
              str(rel.calls))
        rec = slots.cache_record({"cache_n": 10, "prompt_n": 9000,
                                  "prompt_ms": 4321.4, "predicted_n": 300,
                                  "predicted_ms": 6000.0}, None,
                                 {"slot": 0, "mode": "pinned"})
        check(rec.get("prompt_ms") == 4321 and rec.get("decode_tps") == 50.0,
              "the cache record carries the prefill ms and the decode rate "
              "the live test reads", str(rec))
    _with_release(body)


# ---------------------------------------------------------------------------
# RANKS (operator, 2026-09-28): the first conversation never spills; the child
# swaps into VRAM while its conversation's main pauses; a second concurrent
# conversation spills first. Engine patch 0041 moves cells by the ranks each
# request names (kv_rank, kv_ranks); these checks gate what the proxy names.
# ---------------------------------------------------------------------------
def _in_request(key, fn):
    """Run fn() inside a request of conversation `key` (bind_request), as the
    second brain, the decider and a side call run."""
    import cancel
    tok = cancel.Token()
    slots.bind_request(tok, True, "default", [], key=key)
    with cancel.bound(tok):
        return fn()


def _age_act(key: str, seconds: float) -> None:
    """Conversation `key` last active `seconds` ago, nothing in flight."""
    a = slots._act[key]
    a["last"] -= seconds
    a["since"] -= seconds


def _child_rule() -> None:
    """The 2026-09-28 child rule these rank checks gate: the child slot takes
    the rank of the conversation it works for (slots.LANE_RANK off). Layout
    v2's lane ranks above the primary: test_the_lane_ranks_above_the_primary."""
    slots.LANE_RANK = False


def test_ranks_primary_secondary_and_the_child():
    _child_rule()
    slots.reset(n=3)
    a = slots.acquire("conv-a")
    check(a["slot"] == 0 and a["kv_rank"] == slots.RANK_PRIMARY
          and a["kv_ranks"] == [2, 0, 0] and a["primary"] == "conv-a"[:8],
          "the first conversation is PRIMARY: its slot ranks 2, nothing "
          "else ranks", str(a))
    up = slots.upstream_fields(a)
    check(up == {"id_slot": 0, "cache_prompt": True, "kv_rank": 2,
                 "kv_ranks": [2, 0, 0]},
          "what goes upstream: id_slot, cache_prompt and the ranks", str(up))
    slots.release(a)
    # its child while its main pauses (the hidden hop): the primary's rank,
    # and the main's own slot keeps its rank too (equal rank: the engine
    # lets the child take the IDLE main's cells, and the main take them back)
    h = _in_request("conv-a", lambda: slots.acquire(slots.HELPER))
    check(h["slot"] == 2 and h["kv_rank"] == 2 and h["kv_ranks"] == [2, 0, 2],
          "the primary's child ranks as the primary", str(h))
    # a second conversation arrives while the first is live (its child runs)
    b = slots.acquire("conv-b")
    check(b["slot"] == 1 and b["kv_rank"] == slots.RANK_LIVE
          and b["kv_ranks"] == [2, 1, 2],
          "a second concurrent conversation ranks BELOW the first (1): it "
          "spills first", str(b))
    slots.release(h)
    slots.release(b)
    hb = _in_request("conv-b", lambda: slots.acquire(slots.HELPER))
    check(hb["kv_rank"] == 1 and hb["kv_ranks"] == [2, 1, 1],
          "the second conversation's child ranks as the second", str(hb))
    slots.release(hb)
    side = slots.acquire(None, transient=True)
    check(side["slot"] == 2 and side["kv_rank"] == 0
          and side["kv_ranks"] == [2, 1, 0],
          "a side call that works for no conversation ranks 0", str(side))
    slots.release(side)
    w = slots.acquire("conv-a", warm=True)
    check(w["slot"] == 0 and w["kv_rank"] == 2,
          "a warm carries its conversation's rank (a warm is a task: it sets "
          "the slot's rank in the engine)", str(w))
    slots.release(w)
    check(slots._slot_for.get(2) == [] and slots._slot_for.get(0) == []
          and all(v["inflight"] == 0 for v in slots._act.values()),
          "every release takes its request off the books", str(slots._act))


def test_primacy_is_held_between_requests_and_passes_when_the_first_leaves():
    _child_rule()
    slots.reset(n=3)
    slots.release(slots.acquire("conv-a"))
    _age_act("conv-a", slots.PRIMARY_HOLD_S - 5)
    b = slots.acquire("conv-b")
    slots.release(b)
    check(b["kv_rank"] == 1 and b["kv_ranks"] == [2, 1, 0],
          f"the first conversation between two of its requests (idle "
          f"{slots.PRIMARY_HOLD_S - 5:.0f} s < {slots.PRIMARY_HOLD_S:.0f} s) "
          "keeps primacy: its idle cells stay in VRAM", str(b))
    _age_act("conv-a", 10)
    b2 = slots.acquire("conv-b")
    slots.release(b2)
    check(b2["kv_rank"] == 2 and b2["kv_ranks"] == [0, 2, 0],
          "once it has left (idle past PRIMARY_HOLD_S), the live one is "
          "primary and the one that left ranks 0 -- lowered in the engine by "
          "this request's kv_ranks, no request on its slot needed", str(b2))
    a = slots.acquire("conv-a")
    slots.release(a)
    check(a["kv_rank"] == 1 and a["kv_ranks"] == [1, 2, 0],
          "the first comes back while the second is live: it is now the "
          "second (primacy follows activity, it is not taken back)", str(a))
    rf = slots.rank_fields(2)
    check(rf == {"kv_rank": 0, "kv_ranks": [1, 2, 0]},
          "rank_fields: a caller that placed its request itself (the "
          "decider) gets the ranks as they stand", str(rf))
    # a request in flight keeps its conversation live however long it runs
    g = slots.acquire("conv-b")
    _age_act("conv-b", 10 * slots.PRIMARY_HOLD_S)
    x = slots.acquire("conv-a")
    check(x["kv_ranks"] == [1, 2, 0],
          "a long generation keeps its conversation primary", str(x))
    slots.release(x)
    slots.release(g)


def test_a_fork_or_an_adopted_key_takes_over_primacy():
    _child_rule()
    slots.reset(n=3)
    slots.release(slots.acquire("conv-a"))
    f = slots.acquire("conv-fork", inherits="conv-a")
    slots.release(f)
    check(f["kv_rank"] == 2 and f["kv_ranks"] == [1, 2, 0],
          "a fork of the primary (pagoda-h6's post-compaction session) is "
          "primary at once; the parent, still live, ranks below it", str(f))
    slots.reset(n=3)
    slots.release(slots.acquire("conv-a"))
    o = slots.acquire("conv-b", inherits="conv-x")
    slots.release(o)
    check(o["kv_rank"] == 1,
          "`inherits` naming a conversation that is not primary changes "
          "nothing", str(o))
    slots.reset(n=3)
    fp = _fp(USER, CALL, RESULT)
    _turn("conv-a", fp)
    g = slots.acquire("conv-a-renamed", prompt=_fp(
        USER, CALL, RESULT, {"role": "assistant", "content": "done"}))
    slots.release(g)
    check(g.get("adopted") and g["kv_rank"] == 2 and g["kv_ranks"] == [2, 0, 0],
          "an adopted pin (the same conversation under a new key) keeps the "
          "old key's primacy", str(g))


def test_the_rank_switch_and_the_record():
    _child_rule()
    slots.reset(n=3)
    old = slots.KV_RANK
    try:
        slots.KV_RANK = False
        g = slots.acquire("conv-a")
        slots.release(g)
        check("kv_rank" not in g and slots.upstream_fields(g)
              == {"id_slot": 0, "cache_prompt": True}
              and slots.rank_fields(0) == {},
              "YAMADORI_KV_RANK=0: no rank fields go upstream", str(g))
    finally:
        slots.KV_RANK = old
    g = slots.acquire("conv-a")
    slots.release(g)
    rec = slots.cache_record({"cache_n": 1, "prompt_n": 2}, None, g)
    check(rec.get("kv_rank") == 2 and rec.get("kv_ranks") == [2, 0, 0]
          and rec.get("primary") == "conv-a"[:8],
          "x_yamadori.cache records the ranks the engine was told", str(rec))
    snap = slots.snapshot()
    check(snap["child"] == 2 and snap["ranks"]["primary"] == "conv-a"[:8]
          and snap["ranks"]["hold_s"] == slots.PRIMARY_HOLD_S,
          "the snapshot shows the child slot and the primary", str(snap))


# ---------------------------------------------------------------------------
# THE LANE (layout v2, operator 2026-09-29): the child slot serves the decider
# and small side calls, is kept, and ranks above the primary; an as-sent
# compaction goes to the least recently used conversation slot.
# ---------------------------------------------------------------------------
def test_the_lane_ranks_above_the_primary():
    slots.reset(n=3)
    old = (slots.LANE_RANK, slots.LANE_KEEP)
    try:
        slots.LANE_RANK, slots.LANE_KEEP = True, True
        check(slots.RANK_LANE == slots.RANK_PRIMARY + 1 == 3
              and slots.lane_ranked() and slots.lane_kept(),
              "the lane's rank is 3, one above the primary's; ranked and kept "
              "by default")
        a = slots.acquire("conv-a")
        slots.release(a)
        check(a["kv_ranks"] == [2, 0, 3],
              "every request ranks the lane 3, above the primary: the engine "
              "never moves it out of VRAM (0041 plan_moves)", str(a))
        d = _in_request("conv-a", lambda: slots.acquire(None, transient=True))
        check(d["slot"] == 2 and d["kv_rank"] == 3 and d["kv_ranks"] == [2, 0, 3]
              and "lane" in d["how"],
              "the decider or a side call on the lane: rank 3", str(d))
        slots.release(d)
        h = _in_request("conv-a", lambda: slots.acquire(slots.HELPER))
        check(h["slot"] == 2 and h["kv_rank"] == 2 and h["kv_ranks"] == [2, 0, 2],
              "a second-brain request on the lane's slot takes its "
              "conversation's rank (the child rule)", str(h))
        b = slots.acquire("conv-b")
        slots.release(b)
        check(b["kv_ranks"] == [2, 1, 2],
              "while it runs, other requests rank that slot as it", str(b))
        slots.release(h)
        c = slots.acquire("conv-b")
        slots.release(c)
        check(c["kv_ranks"] == [2, 1, 3] and slots.rank_fields(2)
              == {"kv_rank": 3, "kv_ranks": [2, 1, 3]},
              "and back to 3 once it ends; rank_fields (the decider through "
              "model.post) says the same", str(c))
        lg: list = []
        r = slots.lane_kept_note(2, "decider turn", log=lg)
        check(r == lg[0] and r["released"] is False and "kept" in r["skipped"],
              "a kept lane is recorded, not released", str(r))
        snap = slots.snapshot()
        check(snap["lane"] == {"slot": 2, "ranked": True, "rank": 3, "kept": True},
              "the snapshot shows the lane", str(snap.get("lane")))
        slots.LANE_RANK = False
        e = slots.acquire("conv-a")
        slots.release(e)
        check(e["kv_ranks"][2] == 0 and not slots.lane_ranked(),
              "slots.LANE_RANK off: the lane takes its conversation's rank "
              "again (the 2026-09-28 child rule)", str(e))
    finally:
        slots.LANE_RANK, slots.LANE_KEEP = old
        slots.reset(n=3)


COMPACTION = {"role": "user", "content": "You are a summarization agent "
              "creating a context checkpoint. TURNS TO SUMMARIZE: " + "x " * 500}


def test_an_as_sent_compaction_takes_the_lru_conversation_slot():
    slots.reset(n=3)
    fpc = slots.fingerprint({"messages": [COMPACTION]})
    g = slots.acquire(None, transient=True, prefix=fpc)
    slots.release(g)
    check(g["slot"] == 0 and g["mode"] == "compaction" and not g.get("displaced")
          and "no conversation is pinned to" in g["how"],
          "no pins: a free conversation slot, never the lane", str(g))
    slots.reset(n=3)
    _hold("conv-a")
    _hold("conv-b")
    slots._used["conv-a"] -= 60
    g = slots.acquire(None, transient=True, prefix=fpc)
    check(g["slot"] == slots._pins["conv-a"] and g.get("displaced") == "conv-a"[:8]
          and g["mode"] == "compaction" and "least recently used" in g["how"]
          and (g.get("affinity") or {}).get("took") is False,
          "shared with nobody: the least recently used conversation's slot "
          "(a cache miss beats a failed compaction)", str(g))
    slots.release(g)
    s0 = g["slot"]
    check(slots._pins.get("conv-a") == s0 and s0 in slots._dirty
          and (slots._cleared.get(s0) or {}).get("by") == "a compaction sent as is",
          "its pin is kept, the slot is marked for release, and the displaced "
          "conversation's next request will say it resumed cold",
          str((slots._pins, slots._dirty, slots._cleared)))
    nxt = slots.acquire("conv-a")
    slots.release(nxt)
    check(nxt["slot"] == s0 and (nxt.get("resumed_cold") or {}).get("by")
          == "a compaction sent as is",
          "and it does", str(nxt))
    busy = slots.acquire("conv-a")
    busy_b = slots.acquire("conv-b")
    q = slots.acquire(None, transient=True, prefix=fpc)
    check(q["slot"] in (0, 1) and "queued" in q["how"],
          "every conversation slot busy: queued on the least recently used "
          "one, never the lane", str(q))
    for x in (q, busy, busy_b):
        slots.release(x)


def test_a_compaction_slot_is_released_pin_or_not():
    def body(rel):
        _hold("conv-a")
        _hold("conv-b")
        slots._used["conv-a"] -= 60
        fpc = slots.fingerprint({"messages": [COMPACTION]})
        g = slots.acquire(None, transient=True, prefix=fpc)
        slots.release(g)
        r = slots.release_idle(g["slot"], "compaction sent as is ended")
        check(r and r["released"] is False and "pinned" in r["skipped"],
              "without pinned_ok a pinned slot is kept (the rule)", str(r))
        slots._dirty.add(g["slot"])
        r2 = slots.release_idle(g["slot"], "compaction sent as is ended",
                                pinned_ok=True)
        check(r2 and r2["released"] and rel.calls == [g["slot"]],
              "the proxy empties the slot the compaction displaced (pinned_ok)",
              str(r2))
    _with_release(body)


def test_a_stray_pin_on_the_lane_is_dropped():
    slots.reset(n=3)
    slots._pins["old-4-slot-conv"] = 2
    slots._used["old-4-slot-conv"] = 1.0
    g = slots.acquire("old-4-slot-conv")
    slots.release(g)
    check(g["slot"] in (0, 1) and g.get("stray_pins_dropped")
          == [{"key": "old-4-slot"[:8], "slot": 2}],
          "a pin persisted from the four-slot era onto slot 2 (the lane) is "
          "dropped and the conversation re-pinned to a conversation slot "
          "(logs/proxy.log 2026-09-29: 'slot 2 (decider turn) kept -- a "
          "conversation's pinned slot')", str(g))
    check(2 not in slots._pins.values(), "nothing is pinned to the lane",
          str(slots._pins))


def main() -> int:
    for fn in (test_the_helper_slot_is_fixed_and_reserved,
               test_the_second_brain_never_takes_the_conversation_it_serves,
               test_a_busy_helper_slot_defers_rather_than_moving,
               test_transient_calls_spare_conversations,
               test_the_lane_ranks_above_the_primary,
               test_an_as_sent_compaction_takes_the_lru_conversation_slot,
               test_a_compaction_slot_is_released_pin_or_not,
               test_a_stray_pin_on_the_lane_is_dropped,
               test_a_warm_goes_to_the_conversations_own_slot,
               test_a_new_key_adopts_the_slot_its_prompt_continues,
               test_a_shared_harness_head_is_not_enough,
               test_a_busy_or_short_slot_is_not_adopted,
               test_pins_survive_a_restart,
               test_the_helper_slot_is_released_after_a_run_not_between_hops,
               test_never_a_pinned_slot,
               test_never_while_held,
               test_a_new_request_waits_for_a_release_in_flight,
               test_the_switch,
               test_model_release_slot,
               test_an_idle_conversation_is_cleared_for_an_active_one,
               test_idle_is_measured_and_guarded,
               test_a_lone_conversation_never_clears_itself,
               test_the_idle_switch,
               test_the_test_override_touches_only_its_own_account,
               test_ranks_primary_secondary_and_the_child,
               test_primacy_is_held_between_requests_and_passes_when_the_first_leaves,
               test_a_fork_or_an_adopted_key_takes_over_primacy,
               test_the_rank_switch_and_the_record):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
