#!/usr/bin/env python
"""LAYOUT V3 (operator, 2026-09-29: "we should not have a second conversation at all, it is too slow, we have a
second gpu if we want a second conversation"; "We clear jjava lane too after it is done right, not slow down
slop"), offline -- mcp/slots.py ONE CONVERSATION and the lane cleared after each burst.

  1. OFF under today's -np 3: a second conversation gets its own slot, as before.
  2. ON at -np 2: the owner (in flight, or within PRIMARY_HOLD_S of its last request) keeps slot 0; another
     conversation is refused ConversationAtCapacity -> 503 conversation_at_capacity with Retry-After (30 s in
     flight, else the rest of the hold); after the hold the newcomer takes the card and the old pin goes.
  3. COMPACTIONS are never refused for being compactions: the owner's -> slot 0; one sent up as is with no identity
     -> slot 0 once free (waits), bounded by the Retry-After path; never the lane.
  4. A side call -> the lane (slot 1). check_owner() applies the same rule early.
  5. THE LANE: not kept (LANE_KEEP False); a decider burst's release on the lane is recorded "lane burst ended".
No network: model.release_slot is stubbed.
"""
from __future__ import annotations

import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

offline_stores.isolate("yamadori_test_one_conversation_")
os.environ["YAMADORI_SLOTS_STATE"] = os.path.join(os.environ.get("TEMP", "/tmp"), "yamadori_oc_slots.json")

import api_errors  # noqa: E402
import cancel  # noqa: E402
import slots  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok, name, detail=""):
    _results.append((bool(ok), name, str(detail)))


def fresh(n: int) -> None:
    slots.reset(n)
    slots._n = n


def test_off_at_three_slots():
    fresh(3)
    a = slots.acquire("convA")
    b = slots.acquire("convB")
    check(not slots.one_conversation() and {a["slot"], b["slot"]} == {0, 1},
          "-np 3: two conversations on two slots, nothing refused (the rule is off)", (a, b))
    slots.release(a)
    slots.release(b)


def test_owner_and_refusal():
    fresh(2)
    a = slots.acquire("convA")
    check(slots.one_conversation() and a["slot"] == 0 and a.get("one_conversation"), "-np 2: the conversation on 0",
          a)
    try:
        slots.acquire("convB")
        check(False, "a second conversation while the owner is in flight is refused")
    except slots.ConversationAtCapacity as e:
        err = api_errors.of_exception(e)
        check(e.retry_after == 30 and err.status == 503 and err.code == "conversation_at_capacity"
              and err.headers.get("Retry-After") == "30",
              "a second conversation while the owner is in flight: 503 conversation_at_capacity, Retry-After 30",
              (e.retry_after, err.status, err.code))
    slots.release(a)
    try:
        slots.acquire("convB")
        check(False, "within the hold: refused")
    except slots.ConversationAtCapacity as e:
        check(0 < e.retry_after <= slots.PRIMARY_HOLD_S, "within the hold: Retry-After = the rest of it",
              e.retry_after)
    a2 = slots.acquire("convA")
    check(a2["slot"] == 0, "the owner itself is never refused")
    slots.release(a2)
    slots._last_end[0] = time.time() - slots.PRIMARY_HOLD_S - 1
    slots._used["convA"] = time.time() - slots.PRIMARY_HOLD_S - 1      # the hold counts from its latest activity
    b = slots.acquire("convB")
    check(b["slot"] == 0 and "convA" not in slots._pins and slots._pins.get("convB") == 0,
          "after the hold the newcomer takes the card; the old owner's pin goes", (b, dict(slots._pins)))
    slots.release(b)


def test_compactions_and_side_calls():
    fresh(2)
    a = slots.acquire("convA")
    slots.release(a)
    own = slots.acquire("convA", prefix={"x": 1})
    check(own["slot"] == 0 and own["mode"] == "pinned", "the owner's compaction (its key) -> slot 0, its own cache",
          own)
    slots.release(own)
    side = slots.acquire(None, transient=True)
    check(side["slot"] == 1, "a side call -> the lane (slot 1)", side)
    slots.release(side)
    c = slots.acquire(None, transient=True, prefix={"x": 1})
    check(c["slot"] == 0 and c["mode"] == "compaction", "a compaction sent as is on an idle card -> slot 0, never "
          "the lane", c)
    slots.release(c)
    # it waits for a request in flight on slot 0
    a = slots.acquire("convA")
    got: dict = {}

    def comp():
        with cancel.bound(cancel.Token()):
            got["g"] = slots.acquire(None, transient=True, prefix={"x": 1})
    th = threading.Thread(target=comp)
    th.start()
    time.sleep(0.4)
    check(th.is_alive(), "a compaction sent as is waits while the owner's request is in flight (never refused)")
    slots.release(a)
    th.join(5)
    check((got.get("g") or {}).get("slot") == 0, "and takes slot 0 when it frees", got)
    slots.release(got.get("g"))
    # bounded
    a = slots.acquire("convA")
    saved = slots.COMPACTION_WAIT_S
    slots.COMPACTION_WAIT_S = 0.3
    try:
        with cancel.bound(cancel.Token()):
            slots.acquire(None, transient=True, prefix={"x": 1})
        check(False, "the wait is bounded")
    except slots.ConversationAtCapacity as e:
        check(e.retry_after == 30, "the wait is bounded: then the Retry-After path", e.retry_after)
    finally:
        slots.COMPACTION_WAIT_S = saved
        slots.release(a)


def test_check_owner():
    fresh(2)
    a = slots.acquire("convA")
    try:
        slots.check_owner({"key": "convB", "transient": False})
        check(False, "check_owner refuses early")
    except slots.ConversationAtCapacity:
        check(True, "check_owner refuses a second conversation early (before any generation)")
    slots.check_owner({"key": None, "transient": True, "prefix": True})
    slots.check_owner({"key": "convA", "transient": False})
    check(True, "check_owner lets the owner, a side call and an as-sent compaction through")
    slots.release(a)


def test_lane_cleared():
    fresh(2)
    import model
    sent: list = []
    saved = model.release_slot
    model.release_slot = lambda slot, **k: sent.append(slot) or {"ok": True, "ms": 12, "method": "shrink",
                                                                 "cells_before": 3071}
    slots.enable_release(True)
    try:
        check(slots.lane_kept() is False, "the lane is not kept between bursts (operator 2026-09-29)")
        g = slots.acquire(None, transient=True)
        slots.release(g)
        log: list = []
        rec = slots.release_idle(g["slot"], "decider turn", log=log)
        check(sent == [1] and rec and rec.get("released") and rec.get("why") == "lane burst ended"
              and rec.get("by") == "decider turn" and rec.get("ms") == 12 and log,
              "a decider burst's end releases the lane, recorded why 'lane burst ended' with its ms", rec)
    finally:
        model.release_slot = saved
        slots.enable_release(False)


def test_np1_locked_card():
    """-np 1, a LOCKED card (operator, 2026-09-30: Bonsai's jjava and side calls on bonsai-a4000, like Flash-Next):
    the conversation alone on slot 0, a second refused; a compaction still on slot 0 (its own card, its own slot);
    a side call or jjava read bound for the helper server gets that server's LANE slot (1), never a slot of this
    card and never the other card's conversation slot (0)."""
    import max_mode
    saved = (max_mode.ENABLED, max_mode.is_main, max_mode.current, max_mode.decider_model)
    fresh(1)
    try:
        a = slots.acquire("convA")
        check(slots.one_conversation() and a["slot"] == 0, "-np 1: the conversation on slot 0", a)
        try:
            slots.acquire("convB")
            check(False, "-np 1: a second conversation is refused")
        except slots.ConversationAtCapacity:
            check(True, "-np 1: a second conversation is refused")
        slots.release(a)
        c = slots.acquire(None, transient=True, prefix={"x": 1})
        check(c["slot"] == 0 and c["mode"] == "compaction", "-np 1: a compaction sent as is -> slot 0", c)
        slots.release(c)
        max_mode.ENABLED = True
        max_mode.is_main = lambda m: m in ("bonsai", "flash-next", "mirai-s")
        max_mode.decider_model = lambda default=None: "bonsai-a4000"
        max_mode.current = lambda default=None: "bonsai"
        j = slots.acquire(None, transient=True)
        check(j["slot"] == slots.OTHER_LANE_SLOT and j.get("_server") == "bonsai-a4000"
              and j["mode"] == "helper server",
              "a jjava read while bonsai is bound: bonsai-a4000's own slots, none of this card's", j)
        slots.release(j)
        max_mode.current = lambda default=None: "bonsai-a4000"
        sc = slots.acquire(None, transient=True)
        check(sc["slot"] == slots.OTHER_LANE_SLOT and sc["mode"] == "helper server",
              "a side call the table routed to bonsai-a4000: none of this card's slots", sc)
        slots.release(sc)
        max_mode.current = lambda default=None: "bonsai"
        c2 = slots.acquire(None, transient=True, prefix={"x": 1})
        check(c2["slot"] == 0 and c2["mode"] == "compaction",
              "a compaction while jjava is on the helper: still slot 0 of its own card (never the helper)", c2)
        slots.release(c2)
        busy = {s: c for s, c in slots._busy.items() if c}
        check(not busy, "every grant released (a helper-server grant holds nothing here)", busy)
    finally:
        max_mode.ENABLED, max_mode.is_main, max_mode.current, max_mode.decider_model = saved


def _other_card(ok_models=("bonsai",)):
    """Stub the tier table's other card: bonsai's is bonsai-a4000; any other model has none."""
    return lambda m: (("bonsai-a4000", "test table") if m in ok_models else (None, f"{m}: no other card (test)"))


def test_other_card_routing():
    """THE OTHER CARD (operator, 2026-09-30): a conversation whose id is not the main card's owner, while the owner
    is mid-request or inside the 60 s hold (from its LATEST activity), is ROUTED to the other card and keeps it; a
    model with no other card (flash-next, mirai-s) is refused 503 with the rest of the hold -- never downgraded;
    both cards held -> 503; after the hold a new id takes the main card (the switch) and the displaced conversation's
    return records how its prompt came back. Same account or not: no takeover."""
    saved = slots.other_card_for
    slots.other_card_for = _other_card()
    try:
        for n in (1, 2):
            fresh(n)
            wa = {"key": "convA", "account": "acct1"}
            check(slots.check_owner(wa, "bonsai") is None and "routed" not in wa, f"-np {n}: the first conversation "
                  "takes the main card")
            a = slots.acquire("convA")
            # B (the same account, even) while A is in flight: the other card
            wb = {"key": "convB", "account": "acct1"}
            r = slots.check_owner(wb, "bonsai")
            check(r and r["model"] == "bonsai-a4000" and wb["routed"]["routed"] == "other_card"
                  and wb["routed"]["owner_busy"] is True and "owner_idle_s" in wb["routed"],
                  f"-np {n}: a new id while the owner is in flight -> routed to the other card (same account too)",
                  wb)
            b = slots.acquire("convB")
            check(b["slot"] == slots.OTHER_CONV_SLOT and b.get("_server") == "bonsai-a4000" and b["mode"] == "other card"
                  and slots._busy.get(0) == 1 and slots.upstream_fields(b) == {"id_slot": 0, "cache_prompt": True},
                  f"-np {n}: its grant is the other server's conversation slot; the main card's count untouched", b)
            # C at max (flash-next): no other card -> 503, the rest of the hold / 30 s in flight
            wc = {"key": "convC", "account": "acct2"}
            try:
                slots.check_owner(wc, "flash-next")
                check(False, f"-np {n}: a flash-next newcomer is refused")
            except slots.ConversationAtCapacity as e:
                check(e.retry_after == 30 and "does not run on the other card" in str(e),
                      f"-np {n}: a newcomer whose model has no other card (flash-next): 503, never downgraded", str(e))
            # D (bonsai) while both cards are held: 503
            try:
                slots.check_owner({"key": "convD", "account": "acct3"}, "bonsai")
                check(False, f"-np {n}: both cards held -> refused")
            except slots.ConversationAtCapacity as e:
                check("both cards" in str(e), f"-np {n}: both cards held -> 503 conversation_at_capacity", str(e))
            slots.release(a)
            slots.release(b)
            # B keeps its card for its life: even once the main card is free
            slots._last_end[0] = time.time() - slots.PRIMARY_HOLD_S - 1
            slots._last_seen["convA"] = time.time() - slots.PRIMARY_HOLD_S - 1
            slots._used["convA"] = time.time() - slots.PRIMARY_HOLD_S - 1
            wb2 = {"key": "convB", "account": "acct1"}
            r2 = slots.check_owner(wb2, "bonsai")
            check(r2 and r2["model"] == "bonsai-a4000", f"-np {n}: the routed conversation keeps the other card", r2)
            # after the hold a new id takes the main card: the switch
            we = {"key": "convE", "account": "acct4"}
            check(slots.check_owner(we, "bonsai") is None and (we.get("switch") or {}).get("from") == "convA"[:8],
                  f"-np {n}: after the hold a new id takes the main card (want.switch)", we)
            e = slots.acquire("convE")
            check(e["slot"] == 0 and (e.get("switch") or {}).get("from") == "convA"[:8]
                  and "--cache-ram" in e["switch"]["saved"] and "convA" not in slots._pins,
                  f"-np {n}: the switch: A's pin goes, its state to the server's host-RAM prompt cache", e)
            slots.release(e)
            # A comes back after E's hold: takes the card back, and records how its prompt came back
            slots._last_end[0] = time.time() - slots.PRIMARY_HOLD_S - 1
            slots._last_seen["convE"] = time.time() - slots.PRIMARY_HOLD_S - 1
            slots._used["convE"] = time.time() - slots.PRIMARY_HOLD_S - 1
            wa2 = {"key": "convA", "account": "acct1"}
            check(slots.check_owner(wa2, "bonsai") is None, f"-np {n}: A returns after E's hold: the main card")
            a2 = slots.acquire("convA")
            rec = slots.cache_record({"cache_n": 40000, "prompt_n": 20, "prompt_ms": 516, "predicted_ms": 10},
                                     None, a2)
            rc = rec.get("resumed_cold") or {}
            check(rc.get("how") == "restored" and rc.get("prompt_ms") == 516 and "switch" in (rc.get("by") or ""),
                  f"-np {n}: its return records how its prompt came back and the ms (restored, 516 ms)", rc)
            slots.release(a2)
    finally:
        slots.other_card_for = saved


def test_owner_first_race():
    """THE RACE (operator, 2026-09-30: "if there are multiple messages in the queue and they match the id of the
    current session on a card they get the same card again and jump the line"): the owner's request is in flight; a
    different id queues; then the owner's follow-up queues. The owner's follow-up gets the main card next -- its
    arrival also restarts the hold -- and the other id goes to the other card (a Bonsai tier) or is told to retry
    (503: a tier with no other card). One small check, no new queueing machinery (operator: "a rate limit style
    retry after ... should be an edge case or race")."""
    saved = slots.other_card_for
    slots.other_card_for = _other_card()
    try:
        fresh(1)
        slots.check_owner({"key": "own", "account": "a"}, "bonsai")
        first = slots.acquire("own")
        other = {"key": "new", "account": "b"}
        follow = {"key": "own", "account": "a"}
        r_other = slots.check_owner(other, "bonsai")
        r_follow = slots.check_owner(follow, "bonsai")
        f2 = slots.acquire("own")
        check(r_follow is None and f2["slot"] == 0 and "switch" not in follow,
              "the owner's queued follow-up gets the main card next (never displaced by the id that queued first)",
              (follow, f2))
        check(r_other and r_other["model"] == "bonsai-a4000",
              "... and the other id goes to the other card", r_other)
        slots.release(first)
        slots.release(f2)
        # a tier with no other card: told to retry after the rest of the hold, counted from the owner's latest activity
        try:
            slots.check_owner({"key": "late", "account": "c"}, "mirai-s")
            check(False, "a mirai-s newcomer inside the hold is refused")
        except slots.ConversationAtCapacity as e:
            check(slots.PRIMARY_HOLD_S - 2 <= e.retry_after <= slots.PRIMARY_HOLD_S,
                  "a newcomer with no other card: 503 with Retry-After = the rest of the hold from the owner's latest "
                  "activity (its follow-up just now)", e.retry_after)
    finally:
        slots.other_card_for = saved


def test_suite_hold_override():
    """THE LIVE SUITE'S HOLD (slots._hold_for; deploy check 2026-10-01): a request carrying want["hold_s"] (proxy.prepare
    sets it for a TEST account only) shortens the hold only against an owner of the SAME account; a client's owner keeps
    its full hold, and an owner with a request in flight holds the card whatever the override says."""
    saved = slots.other_card_for
    slots.other_card_for = _other_card()
    try:
        fresh(1)
        slots.check_owner({"key": "t1", "account": "test"}, "bonsai")
        a = slots.acquire("t1")
        w = {"key": "t2", "account": "test", "hold_s": 0}
        r = slots.check_owner(w, "bonsai")
        check(r and r["model"] == "bonsai-a4000", "the override never takes a card with a request in flight", r)
        slots.release(a)
        slots._other.update(owner=None, model=None)
        w = {"key": "t3", "account": "test", "hold_s": 0}
        check(slots.check_owner(w, "bonsai") is None and (w.get("switch") or {}).get("from") == "t1",
              "hold 0 against its own account's finished owner: the newcomer takes the main card", w)
        t3 = slots.acquire("t3")
        check(t3["slot"] == 0, "... and acquire grants it slot 0 (the backstop agrees)", t3)
        slots.release(t3)
        # a client's owner: its full hold, whatever the test sends
        fresh(1)
        slots.check_owner({"key": "client", "account": "operator"}, "bonsai")
        slots.release(slots.acquire("client"))
        w = {"key": "t4", "account": "test", "hold_s": 0}
        r = slots.check_owner(w, "bonsai")
        check(r and r["model"] == "bonsai-a4000" and "60 s hold" in r["why"],
              "a client's owner keeps the full hold against a test override (the newcomer is routed)", r)
        w = {"key": "t5", "account": "test", "hold_s": 0}
        try:
            slots.check_owner(w, "mirai-s")
            check(False, "a test newcomer with no other card inside a client's hold is refused")
        except slots.ConversationAtCapacity as e:
            check(e.retry_after > 1, "... and one with no other card is told 503 with the rest of the client's hold",
                  e.retry_after)
    finally:
        slots.other_card_for = saved


def main() -> int:
    for fn in (test_off_at_three_slots, test_owner_and_refusal, test_compactions_and_side_calls, test_check_owner,
               test_lane_cleared, test_np1_locked_card, test_other_card_routing,
               test_owner_first_race, test_suite_hold_override):
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
            import traceback
            check(False, f"{fn.__name__} raised", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}")
    bad = [r for r in _results if not r[0]]
    for ok, name, detail in _results:
        print(("ok    " if ok else "FAIL  ") + name + (f"  ({detail[:400]})" if not ok and detail else ""))
    print(f"\n{'=' * 70}\n  {len(_results) - len(bad)}/{len(_results)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
