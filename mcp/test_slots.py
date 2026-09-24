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

The rule now: helper_slot() = n-2 is the second brain's in every process, and
no conversation is ever pinned to it.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["YAMADORI_SLOTS"] = "4"

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
    slots.reset(n=4)
    check(slots.helper_slot(4) == 2 and slots.helper_slot(3) == 1
          and slots.helper_slot(2) is None,
          "helper_slot is n-2 (slot 2 of 4), the same number in every "
          "process; none below three slots")
    got = [_hold(f"conv-{i}")["slot"] for i in range(5)]
    check(2 not in got and 3 not in got,
          "five conversations in turn are never pinned to the helper slot "
          "(2) or the transient slot (3)", str(got))
    check(set(got) == {0, 1}, "they share slots 0 and 1 by LRU", str(got))
    g = slots.acquire(slots.HELPER)
    slots.release(g)
    check(g["slot"] == 2 and g["mode"] == "helper" and not g["evicted"],
          "the second brain gets slot 2 and evicts nobody", str(g))
    check(slots.HELPER not in slots._pins,
          "the second brain holds no pin (nothing can evict it, it can "
          "evict nothing)", str(slots._pins))


def test_the_second_brain_never_takes_the_conversation_it_serves():
    """The 12:49 shape: every conversation slot pinned, and the conversation
    being served is the least recently used (its last turn was long ago);
    deep thinking runs before its turn acquires anything."""
    slots.reset(n=4)
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
          "all 13 deep-thinking hops run on the helper slot", str(hops))
    check(slots._pins == before,
          "no conversation lost its pin to the second brain",
          f"{before} -> {slots._pins}")
    g = slots.acquire("served")
    slots.release(g)
    check(g["slot"] == served_slot and not g["evicted"],
          "the served conversation's turn then finds its own slot "
          "(its cached prefix intact)", str(g))


def test_a_busy_helper_slot_defers_rather_than_moving():
    slots.reset(n=4)
    _hold("c0")
    _hold("c1")
    a = slots.acquire(slots.HELPER)
    b = slots.acquire(slots.HELPER)       # e.g. a summary during fan-out
    check(a["slot"] == b["slot"] == 2,
          "a second concurrent second-brain call still targets slot 2 (the "
          "server defers it) instead of moving onto a conversation's slot",
          f"{a} {b}")
    slots.release(a)
    slots.release(b)


def test_transient_calls_spare_conversations():
    slots.reset(n=4)
    _hold("c0")
    _hold("c1")
    t = slots.acquire(None, transient=True)
    check(t["slot"] == 3, "a side call takes the transient slot", str(t))
    t2 = slots.acquire(None, transient=True)
    check(t2["slot"] == 2 and not t2["evicted"],
          "with the transient slot busy, the next takes the IDLE helper slot "
          "before evicting a conversation", str(t2))
    slots.release(t)
    slots.release(t2)
    check(slots._pins.get("c0") is not None
          and slots._pins.get("c1") is not None,
          "both conversations keep their pins", str(slots._pins))


def test_a_warm_goes_to_the_conversations_own_slot():
    slots.reset(n=4)
    g = _hold("conv")
    w = slots.acquire("conv", warm=True)
    slots.release(w)
    check(w["slot"] == g["slot"] and w["mode"] == "warm",
          "a warm targets the slot the conversation generated on",
          f"{g} {w}")


def main() -> int:
    for fn in (test_the_helper_slot_is_fixed_and_reserved,
               test_the_second_brain_never_takes_the_conversation_it_serves,
               test_a_busy_helper_slot_defers_rather_than_moving,
               test_transient_calls_spare_conversations,
               test_a_warm_goes_to_the_conversations_own_slot):
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
