#!/usr/bin/env python
"""Admission control, asserted with real semaphores and no model.

WHAT THIS IS GATING

`admission.py` is the cap that stopped three benchmark processes degrading
each other into unattributable 502s. Its promises:

  1. At most MAIN_LANES main requests hold a lane at once; the next one waits
     WAIT_SECONDS and is then REFUSED with `Full`, which says what happened
     and that retrying can work -- never queued forever.
  2. Helpers have their own lanes -- HELPER_LANES = 1, one second brain
     with 3/8 of the pool (mcp/budget.py, the operator's split of
     2026-09-22) -- so a full main pool never blocks the investigation its
     own conversation is waiting on (that is a deadlock), and the helper
     past HELPER_LANES is refused. The helper tests below are written over
     admission.HELPER_LANES, so they hold for whatever count ships.
  3. A lane is always given back: on exit, on exception, and after a refused
     wait. A leaked permit is a server that slowly admits nobody.
  4. `helper_lane` -- the threading semaphore around the second contexts --
     admits HELPER_LANES, returns False for the next one instead of running
     anyway, and never releases a lane it did not get.

WHAT IS COVERED ELSEWHERE

`mcp/test_tools.py` stubs `helper_lane` to assert the PROXY reports a refused
lane as HELPER_BUSY. This file tests the lane itself, which that one does not.

NO NETWORK

`snapshot()` asks `budget` for the pool, which would ask the live server.
`budget._POOL` is pinned and `urlopen` is blocked before anything runs.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import traceback
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

for k in ("YAMADORI_MAIN_LANES", "YAMADORI_HELPER_LANES", "YAMADORI_ADMIT_WAIT"):
    os.environ.pop(k, None)

import admission  # noqa: E402
import budget  # noqa: E402

_leaks: list[str] = []


def _blocked(url, timeout=None):                                  # noqa: ARG001
    _leaks.append(str(getattr(url, "full_url", url)))
    raise OSError("test_admission: network is blocked")


urllib.request.urlopen = _blocked
budget._POOL = 147456

# Short enough to keep the suite fast, long enough that a lane released
# mid-wait is seen.
admission.WAIT_SECONDS = 0.3

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def fresh() -> None:
    """New semaphores and counters. asyncio semaphores bind to the loop that
    first waits on them, and each test runs its own loop."""
    admission._lanes.clear()
    for k in admission._stats:
        admission._stats[k] = 0
    for k in admission._inflight:
        admission._inflight[k] = 0


# ---------------------------------------------------------------------------


def test_the_fixture_is_offline_and_shipped():
    check(admission.MAIN_LANES == 2 and admission.HELPER_LANES == 1,
          "the shipped lane counts are under test", f"{admission.MAIN_LANES}/"
          f"{admission.HELPER_LANES}")
    check(budget._POOL == 147456, "the pool is pinned, so nothing asks a server")


def test_two_main_requests_run_and_the_third_is_refused():
    fresh()

    async def go():
        a, b = admission.admit("main"), admission.admit("main")
        await a.__aenter__()
        await b.__aenter__()
        inflight = admission._inflight["main"]
        t0 = time.time()
        try:
            async with admission.admit("main"):
                third = "admitted"
        except admission.Full as e:
            third = e
        waited = time.time() - t0
        await a.__aexit__(None, None, None)
        await b.__aexit__(None, None, None)
        return inflight, third, waited

    inflight, third, waited = asyncio.run(go())
    check(inflight == 2, "two main requests hold lanes at once", str(inflight))
    check(isinstance(third, admission.Full), "the third is refused with Full",
          repr(third))
    check(waited >= admission.WAIT_SECONDS * 0.9,
          "after waiting its turn, not immediately", f"{waited:.2f}s")
    check(waited < admission.WAIT_SECONDS + 1.0, "and not indefinitely",
          f"{waited:.2f}s")
    if isinstance(third, admission.Full):
        msg = str(third)
        check(third.kind == "main" and third.waited > 0,
              "Full carries the lane kind and how long it waited")
        check("all 2 main lanes busy" in msg, "the message names the situation", msg)
        check("Retry" in msg, "and says retrying can work", msg)
    s = admission._stats
    check(s["admitted"] == 2 and s["refused"] == 1 and s["queued"] == 1,
          "the counters record 2 admitted, 1 queued, 1 refused", str(s))
    check(s["peak_main"] == 2, "and the peak", str(s["peak_main"]))


def test_a_refused_wait_leaks_no_lane():
    fresh()

    async def go():
        holders = [admission.admit("main") for _ in range(2)]
        for h in holders:
            await h.__aenter__()
        for _ in range(3):
            try:
                async with admission.admit("main"):
                    pass
            except admission.Full:
                pass
        for h in holders:
            await h.__aexit__(None, None, None)
        # Every lane must be free again: two more enter without waiting.
        t0 = time.time()
        again = [admission.admit("main") for _ in range(2)]
        for h in again:
            await h.__aenter__()
        took = time.time() - t0
        for h in again:
            await h.__aexit__(None, None, None)
        return took

    took = asyncio.run(go())
    check(took < 0.1, "after three refusals both lanes are free at once",
          f"{took:.2f}s")
    check(admission._inflight["main"] == 0, "and nothing is counted in flight",
          str(admission._inflight))


def test_a_waiter_is_admitted_when_a_lane_frees_in_time():
    fresh()

    async def go():
        a, b = admission.admit("main"), admission.admit("main")
        await a.__aenter__()
        await b.__aenter__()

        async def release_soon():
            await asyncio.sleep(admission.WAIT_SECONDS / 3)
            await a.__aexit__(None, None, None)
        asyncio.get_running_loop().create_task(release_soon())
        try:
            async with admission.admit("main"):
                got = True
        except admission.Full:
            got = False
        await b.__aexit__(None, None, None)
        return got

    check(asyncio.run(go()), "a request queued behind a finishing one gets in")


def test_helpers_have_their_own_lane():
    fresh()

    async def go():
        mains = [admission.admit("main") for _ in range(2)]
        for m in mains:
            await m.__aenter__()
        try:
            async with admission.admit("helper"):
                helper = True
        except admission.Full:
            helper = False
        held = [admission.admit("helper") for _ in range(n)]
        for h in held:
            await h.__aenter__()
        inflight = admission._inflight["helper"]
        try:
            async with admission.admit("helper"):
                extra = True
        except admission.Full as e:
            extra = e
        for h in held:
            await h.__aexit__(None, None, None)
        for m in mains:
            await m.__aexit__(None, None, None)
        return helper, inflight, extra

    n = admission.HELPER_LANES
    helper, inflight, extra = asyncio.run(go())
    check(helper, "a helper is admitted while every main lane is full")
    check(inflight == n, f"{n} helper(s) hold lanes at once (HELPER_LANES)",
          str(inflight))
    check(isinstance(extra, admission.Full) and extra.kind == "helper",
          f"and helper number {n + 1} is refused on the helper lane",
          repr(extra))
    if isinstance(extra, admission.Full):
        check(f"all {n} helper lanes busy" in str(extra),
              "naming the helper lane count", str(extra))


def test_an_unknown_kind_is_a_main_request():
    check(admission.admit("banana").kind == "main", "an unknown kind uses a main lane")


def test_an_exception_inside_releases_the_lane():
    fresh()

    async def go():
        for _ in range(5):
            try:
                async with admission.admit("main"):
                    raise ValueError("handler blew up")
            except ValueError:
                pass
        return admission._inflight["main"], admission._sem("main")._value

    inflight, free = asyncio.run(go())
    check(inflight == 0 and free == admission.MAIN_LANES,
          "five failing requests leave every lane free", f"{inflight}, {free}")


def test_helper_lane_refuses_instead_of_running_anyway():
    for k in admission._helper_stats:
        admission._helper_stats[k] = 0
    n = admission.HELPER_LANES
    held = [admission.helper_lane(timeout=0.1) for _ in range(n)]
    got = [h.__enter__() for h in held]
    inflight = admission._helper_stats["inflight"]
    extra = admission.helper_lane(timeout=0.1)
    t0 = time.time()
    got_extra = extra.__enter__()
    waited = time.time() - t0
    extra.__exit__(None, None, None)
    # The refused one must not have released a lane it never held.
    again = admission.helper_lane(timeout=0.05)
    got_again = again.__enter__()
    again.__exit__(None, None, None)
    for h in reversed(held):
        h.__exit__(None, None, None)
    check(all(g is True for g in got),
          f"{n} investigation(s) get a lane each (HELPER_LANES)", str(got))
    check(inflight == n, f"and all {n} are counted in flight", str(inflight))
    check(got_extra is False,
          f"investigation number {n + 1} is refused (False), not run alongside")
    check(waited >= 0.09, "after waiting its timeout", f"{waited:.2f}s")
    check(got_again is False,
          "a refused lane's exit releases nothing it did not hold")
    s = admission._helper_stats
    check(s["granted"] == n and s["timed_out"] == 2 and s["inflight"] == 0,
          f"stats: {n} granted, 2 timed out, none in flight", str(s))
    with admission.helper_lane(timeout=0.05) as ok:
        check(ok is True, "after release, the lane is free again")


def test_helper_lane_works_across_threads():
    box: dict = {}
    held = [admission.helper_lane(timeout=0.1)
            for _ in range(admission.HELPER_LANES)]
    for h in held:
        h.__enter__()
    first, rest = held[0], held[1:]

    def other():
        with admission.helper_lane(timeout=1.0) as ok:
            box["ok"] = ok
    t = threading.Thread(target=other)
    t.start()
    time.sleep(0.1)
    first.__exit__(None, None, None)
    t.join(timeout=2)
    for h in rest:
        h.__exit__(None, None, None)
    check(box.get("ok") is True,
          "an investigation on another thread gets the lane once it is released")


def test_snapshot_reports_the_cap_and_the_ceiling():
    fresh()
    snap = admission.snapshot()
    check(snap["main_lanes"] == admission.MAIN_LANES
          and snap["helper_lanes"] == admission.HELPER_LANES,
          "the snapshot reports the lane counts",
          f"{snap['main_lanes']}/{snap['helper_lanes']}")
    check(snap["pool"] == 147456 and snap["main_budget"] == 92160,
          "and the budget it derives from (5/8 of the pool)",
          f"{snap['pool']}/{snap['main_budget']}")
    check(snap["full_budget_ceiling"] == 1,
          "ONE main request at full budget fits the pool (92,160 x 2 would "
          "not), so MAIN_LANES = 2 is stated oversubscription",
          str(snap["full_budget_ceiling"]))
    old = budget.budgets
    try:
        def broken(*a, **k):
            raise RuntimeError("budget unavailable")
        budget.budgets = broken
        snap = admission.snapshot()
        check(snap["pool"] is None and snap["full_budget_ceiling"] is None,
              "an unavailable budget is reported as unknown, not zero")
    finally:
        budget.budgets = old


def main() -> int:
    for fn in (test_the_fixture_is_offline_and_shipped,
               test_two_main_requests_run_and_the_third_is_refused,
               test_a_refused_wait_leaks_no_lane,
               test_a_waiter_is_admitted_when_a_lane_frees_in_time,
               test_helpers_have_their_own_lane,
               test_an_unknown_kind_is_a_main_request,
               test_an_exception_inside_releases_the_lane,
               test_helper_lane_refuses_instead_of_running_anyway,
               test_helper_lane_works_across_threads,
               test_snapshot_reports_the_cap_and_the_ceiling):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    check(not _leaks, "no test reached a real server", str(_leaks))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
