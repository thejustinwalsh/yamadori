#!/usr/bin/env python
"""The KV-pool split, asserted. No network, no model server.

WHAT THIS IS GATING

`budget.py` decides how many tokens the conversation and its helpers may send
into ONE unified KV pool. The numbers in its docstring are the contract:

  pool 147,456 -> main 92,160 + 1 helper x 55,296 + reserve 0, summing exactly

(the operator's split of 2026-09-22: 5/8 for the conversation, 3/8 for ONE
second brain. It replaced, the same day, 1/2 + 2 x 1/4 -- main 73,728 and
36,864 for each of up to two helpers -- whose second helper needed two
conversations investigating at once. That in turn replaced 60/25/15 -- main
88,473, one helper, reserve 22,119 -- which had no measurement behind it; see
docs/CONSTRAINTS.md item 19.)

and three properties that must hold however the shares are configured:

  - main never drops below its floor (a helper starving its own
    conversation is the one outcome worse than no split at all)
  - main + HELPERS x helper never exceed the pool; when the shares overflow,
    the helper is cut to (pool - main) // HELPERS
  - main + HELPERS x helper + reserve always sum to the pool

NO NETWORK

`pool_size()` asks the live server for its n_ctx. Every test either passes an
explicit pool or replaces `urllib.request.urlopen` with a stub, so nothing here
can reach :10001 or :11434 -- the first test asserts that the stub is what is
being called.
"""
from __future__ import annotations

import io
import json
import os
import sys
import traceback
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import budget  # noqa: E402

SHIPPED_POOL = 147456   # config.yaml, `-c 147456`

_results: list[tuple[bool, str, str]] = []
_calls: list[str] = []
_leaks: list[str] = []


def _blocked(url, timeout=None):                                  # noqa: ARG001
    """Installed whenever no test has asked for a stub. A real request that
    got this far would have reached the live server; it is recorded and
    refused instead, and the last check fails if there were any."""
    _leaks.append(url if isinstance(url, str) else url.full_url)
    raise OSError("test_budget: network is blocked")


urllib.request.urlopen = _blocked


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def fake_server(payload_by_url: dict | None):
    """Replace urlopen. `None` means every URL is unreachable."""
    def urlopen(url, timeout=None):                               # noqa: ARG001
        u = url if isinstance(url, str) else url.full_url
        _calls.append(u)
        if payload_by_url is None or u not in payload_by_url:
            raise OSError(f"stub: {u} unreachable")
        return io.BytesIO(json.dumps(payload_by_url[u]).encode())
    urllib.request.urlopen = urlopen
    budget._POOL = None


def restore() -> None:
    urllib.request.urlopen = _blocked


def with_shares(main: float, helper: float, floor: float,
                helpers: int | None = None):
    """Set the shares (and optionally the helper count); returns what to pass
    back to restore them."""
    old = (budget.MAIN_SHARE, budget.HELPER_SHARE, budget.MAIN_FLOOR,
           budget.HELPERS)
    budget.MAIN_SHARE, budget.HELPER_SHARE, budget.MAIN_FLOOR = main, helper, floor
    if helpers is not None:
        budget.HELPERS = helpers
    return old


# ---------------------------------------------------------------------------


def test_the_fixture_cannot_reach_a_server():
    fake_server(None)
    try:
        _calls.clear()
        p = budget.pool_size(refresh=True)
        check(len(_calls) == 2, "pool_size asked the stub, twice (direct, then "
              "upstream)", str(_calls))
        check(all(u.endswith("/props") for u in _calls),
              "and asked for /props", str(_calls))
        check(p == 131072, "unreachable falls back to 131072 (documented, "
              "deliberately left)", str(p))
    finally:
        restore()


def test_the_shipped_split_is_the_documented_one():
    check((budget.MAIN_SHARE, budget.HELPER_SHARE, budget.HELPERS,
           budget.MAIN_FLOOR) == (0.625, 0.375, 1, 0.50),
          "the shipped shares are 5/8 main, 3/8 for one helper, floor 1/2",
          str((budget.MAIN_SHARE, budget.HELPER_SHARE,
                            budget.HELPERS, budget.MAIN_FLOOR)))
    b = budget.budgets(SHIPPED_POOL)
    check(set(b) == {"pool", "main", "helper", "helpers", "reserve", "gib"},
          "budgets() reports pool, main, helper, helpers, reserve, gib",
          str(sorted(b)))
    check(b["pool"] == SHIPPED_POOL, "the pool is the one passed in", str(b["pool"]))
    check(b["main"] == 92160, "main is 92,160 at the shipped pool", str(b["main"]))
    check(b["helper"] == 55296, "the helper is 55,296", str(b["helper"]))
    check(b["helpers"] == 1, "there is one helper", str(b["helpers"]))
    check(b["reserve"] == 0, "nothing is left unclaimed (reserve 0)",
          str(b["reserve"]))
    check(b["main"] + b["helpers"] * b["helper"] + b["reserve"] == SHIPPED_POOL,
          "and main + 1 x helper + reserve sum to the pool exactly", json.dumps(b))
    check(b["gib"] == 6.19, "the KV at 44 KiB/token is 6.19 GiB, as documented",
          str(b["gib"]))
    b = budget.budgets(163840)
    check((b["main"], b["helper"], b["reserve"]) == (102400, 61440, 0),
          "at the live 163,840 (config.yaml -c) it is 102,400 + 61,440, "
          "reserve 0", json.dumps(b))


def test_the_invariants_hold_for_any_shares():
    cases = [(0.625, 0.375, 0.50), (0.50, 0.25, 0.50), (0.60, 0.25, 0.50),
             (0.30, 0.25, 0.50), (0.60, 0.60, 0.50), (0.50, 0.30, 0.50),
             (0.90, 0.90, 0.50), (0.10, 0.10, 0.80), (1.0, 0.5, 0.5),
             (0.0, 1.0, 0.0)]
    counts = (1, 2, 3)
    for helpers in counts:
        for main, helper, floor in cases:
            old = with_shares(main, helper, floor, helpers)
            try:
                for pool in (1, 8192, 131072, SHIPPED_POOL, 262144):
                    b = budget.budgets(pool)
                    tag = (f"shares {main}/{helper} x{helpers} floor {floor}, "
                           f"pool {pool}")
                    n = b["helpers"]
                    ok = (n == helpers
                          and b["main"] >= int(pool * floor)
                          and b["main"] + n * b["helper"] <= pool
                          and b["helper"] >= 0 and b["reserve"] >= 0
                          and b["main"] + n * b["helper"] + b["reserve"] == pool)
                    if not ok:
                        check(False, "floor, ceiling and sum all hold",
                              f"{tag}: {b}")
                        return
            finally:
                with_shares(*old)
    check(True, f"floor, ceiling and sum hold across {len(cases)} share "
          f"settings x {len(counts)} helper counts x 5 pool sizes")


def test_the_floor_protects_the_conversation():
    # With the shipped single helper.
    old = with_shares(0.30, 0.60, 0.50, 1)
    try:
        b = budget.budgets(100000)
        check(b["main"] == 50000, "a 30% main share is raised to the 50% floor",
              str(b["main"]))
        check(b["helper"] == 50000,
              "and the one helper is cut to (pool - main) // 1, not given its "
              "60%", str(b["helper"]))
        check(b["reserve"] == 0, "which leaves nothing over", str(b["reserve"]))
    finally:
        with_shares(*old)
    old = with_shares(0.625, 0.45, 0.50, 1)
    try:
        b = budget.budgets(SHIPPED_POOL)
        check(b["main"] == 92160 and b["helper"] == 55296,
              "a helper at 45% would overflow 5/8 + 45%, so it is cut to "
              "55,296, not 66,355", json.dumps(b))
    finally:
        with_shares(*old)
    old = with_shares(0.625, 0.25, 0.50, 1)
    try:
        b = budget.budgets(100000)
        check(b["helper"] == 25000 and b["reserve"] == 12500,
              "shares that fit are left alone and the rest is reserve",
              json.dumps(b))
    finally:
        with_shares(*old)
    # With several helpers the cut divides the room: (pool - main) // HELPERS.
    old = with_shares(0.30, 0.60, 0.50, 2)
    try:
        b = budget.budgets(100000)
        check(b["main"] == 50000 and b["helper"] == 25000 and b["reserve"] == 0,
              "with HELPERS=2, each helper is cut to (pool - main) // 2",
              json.dumps(b))
    finally:
        with_shares(*old)
    old = with_shares(0.50, 0.30, 0.50, 2)
    try:
        b = budget.budgets(SHIPPED_POOL)
        check(b["main"] == 73728 and b["helper"] == 36864,
              "with HELPERS=2, helpers at 30% would overflow 2 x 30% + 50%, so "
              "each is cut to 36,864, not 44,236", json.dumps(b))
    finally:
        with_shares(*old)
    check(budget.HELPERS == 1, "and the helper count is restored after",
          str(budget.HELPERS))


def test_pool_size_reads_the_server_and_caches():
    fake_server({f"{budget.DIRECT}/props":
                 {"default_generation_settings": {"n_ctx": SHIPPED_POOL}}})
    try:
        _calls.clear()
        check(budget.pool_size(refresh=True) == SHIPPED_POOL,
              "the pool comes from default_generation_settings.n_ctx")
        budget.pool_size()
        check(len(_calls) == 1, "a second call is served from the cache",
              str(len(_calls)))
    finally:
        restore()

    fake_server({f"{budget.UPSTREAM}/props": {"n_ctx": 65536}})
    try:
        check(budget.pool_size(refresh=True) == 65536,
              "a top-level n_ctx on the upstream is used when direct is down")
    finally:
        restore()

    fake_server({f"{budget.DIRECT}/props": {"n_ctx": 0},
                 f"{budget.UPSTREAM}/props": {"unrelated": 1}})
    try:
        check(budget.pool_size(refresh=True) == 131072,
              "a server that reports no usable n_ctx falls back, not zero")
    finally:
        restore()


def test_cap_for_answers_only_main_or_helper():
    budget._POOL = SHIPPED_POOL
    check(budget.cap_for("main") == 92160, "cap_for('main') is the main budget")
    check(budget.cap_for("helper") == 55296,
          "cap_for('helper') is the helper's budget")
    for other in ("pool", "gib", "reserve", "helpers", "nonsense", ""):
        v = budget.cap_for(other)
        check(v == 92160, f"cap_for({other!r}) falls back to main, not a "
              f"report field", repr(v))


def test_fits_is_inclusive_and_explains_a_refusal():
    budget._POOL = SHIPPED_POOL
    ok, why = budget.fits(92160, "main")
    check(ok and why == "", "exactly the budget fits")
    ok, why = budget.fits(92161, "main")
    check(not ok, "one token over does not")
    check("92161" in why and "92160" in why and "147456" in why,
          "the refusal names the size, the budget and the pool", why)
    check("raise -c" in why or "Trim" in why, "and says what to do", why)
    ok, _ = budget.fits(60000, "helper")
    check(not ok, "a helper is held to the helper budget, not main's")
    ok, _ = budget.fits(55296, "helper")
    check(ok, "and exactly the helper budget fits")


def test_what_if_includes_the_live_pool():
    """The default table used to list every candidate EXCEPT the one running."""
    table = budget.what_if()
    rows = [ln.split() for ln in table.splitlines()[1:]]
    pools = [int(r[0]) for r in rows]
    check(SHIPPED_POOL in pools, "what_if() has a row for 147456", str(pools))
    check(pools == sorted(pools), "rows are in ascending pool order", str(pools))
    live = next((r for r in rows if int(r[0]) == SHIPPED_POOL), None)
    if live:
        check(live[2:5] == ["92160", "55296", "0"],
              "and that row shows the documented split", " ".join(live))
        check(live[5] == "yes", "and says it fits the 16.3 GB card", " ".join(live))
    big = next((r for r in rows if int(r[0]) == 262144), None)
    check(big is not None and big[5] == "NO",
          "262144 is reported as NOT fitting, as documented",
          " ".join(big) if big else "missing")


def main() -> int:
    for fn in (test_the_fixture_cannot_reach_a_server,
               test_the_shipped_split_is_the_documented_one,
               test_the_invariants_hold_for_any_shares,
               test_the_floor_protects_the_conversation,
               test_pool_size_reads_the_server_and_caches,
               test_cap_for_answers_only_main_or_helper,
               test_fits_is_inclusive_and_explains_a_refusal,
               test_what_if_includes_the_live_pool):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        finally:
            restore()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    check(not _leaks, "no test reached for a real server", str(_leaks))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
