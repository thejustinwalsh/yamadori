#!/usr/bin/env python
"""The KV-pool split, asserted. No network, no model server.

WHAT THIS IS GATING

`budget.py` decides how many tokens the conversation and its helpers may send
into ONE unified KV pool. The numbers in its docstring are the contract:

  pool 147,456 -> main 92,160 + 1 helper x 55,296 + reserve 0, summing exactly

(the operator's split of 2026-09-25: 0.70 for the conversation, 0.30 for ONE
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

SHIPPED_POOL = 147456   # the pool the mechanics tests were written at
LIVE_POOL = 163840      # config.yaml `-c 163840` (q8_0, 2026-09-25)

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
                helpers: int | None = None, helper_tokens: int = 0):
    """Set the shares (and optionally the helper count); returns what to pass
    back to restore them. The fixed helper size is off unless given, so the
    fractions are what is tested."""
    old = (budget.MAIN_SHARE, budget.HELPER_SHARE, budget.MAIN_FLOOR,
           budget.HELPERS, budget.HELPER_TOKENS)
    budget.MAIN_SHARE, budget.HELPER_SHARE, budget.MAIN_FLOOR = main, helper, floor
    budget.HELPER_TOKENS = helper_tokens
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
    check(budget.HELPER_TOKENS == 49152,
          "the shipped helper is a fixed 49,152 tokens (main gets the rest)",
          str(budget.HELPER_TOKENS))
    b = budget.budgets(181248)
    check((b["main"], b["helper"], b["reserve"]) == (132096, 49152, 0),
          "at -c 181,248 the new tokens go to main: 132,096 + 49,152",
          json.dumps(b))
    check((budget.MAIN_SHARE, budget.HELPER_SHARE, budget.HELPERS,
           budget.MAIN_FLOOR) == (0.70, 0.30, 1, 0.50),
          "the fallback shares are 0.70 main, 0.30 for one helper, floor 1/2",
          str((budget.MAIN_SHARE, budget.HELPER_SHARE,
                            budget.HELPERS, budget.MAIN_FLOOR)))
    b = budget.budgets(LIVE_POOL)
    check(set(b) == {"pool", "main", "helper", "helpers", "reserve", "gib",
                     "window", "layout", "cap_source", "child"}
          and b["child"]["role"] == "decider lane"
          and b["child"]["tokens"] == budget.LANE_TOKENS,
          "budgets() reports pool, main, helper, helpers, reserve, gib, the "
          "child slot's role (layout v2: the decider lane), and "
          "the window, layout and cap source",
          str(sorted(b)))
    check(b["layout"] == "split" and b["window"] == LIVE_POOL,
          "with no cap known (no served line, no YAMADORI_MAIN_CAP) the split "
          "applies and one request may use the pool", str(b))
    check(b["pool"] == LIVE_POOL, "the pool is the one passed in", str(b["pool"]))
    check(b["main"] == 114688, "main is 114,688 at the shipped pool", str(b["main"]))
    check(b["helper"] == 49152, "the helper is 49,152", str(b["helper"]))
    check(b["helpers"] == 1, "there is one helper", str(b["helpers"]))
    check(b["reserve"] == 0, "nothing is left unclaimed (reserve 0)",
          str(b["reserve"]))
    check(b["main"] + b["helpers"] * b["helper"] + b["reserve"] == LIVE_POOL,
          "and main + 1 x helper + reserve sum to the pool exactly", json.dumps(b))
    check(b["gib"] == 6.88, "the KV at 44 KiB/token is 6.88 GiB at 163,840",
          str(b["gib"]))
    b = budget.budgets(163840)
    check((b["main"], b["helper"], b["reserve"]) == (114688, 49152, 0),
          "at the live 163,840 (config.yaml -c) it is 114,688 + 49,152, "
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
    budget._POOL = LIVE_POOL
    check(budget.cap_for("main") == 114688, "cap_for('main') is the main budget")
    check(budget.cap_for("helper") == 49152,
          "cap_for('helper') is the helper's budget")
    for other in ("pool", "gib", "reserve", "helpers", "nonsense", ""):
        v = budget.cap_for(other)
        check(v == 114688, f"cap_for({other!r}) falls back to main, not a "
              f"report field", repr(v))


def test_fits_is_inclusive_and_explains_a_refusal():
    budget._POOL = LIVE_POOL
    ok, why = budget.fits(114688, "main")
    check(ok and why == "", "exactly the budget fits")
    ok, why = budget.fits(114689, "main")
    check(not ok, "one token over does not")
    check("114689" in why and "114688" in why and "163840" in why,
          "the refusal names the size, the budget and the pool", why)
    check("raise -c" in why or "Trim" in why, "and says what to do", why)
    ok, _ = budget.fits(60000, "helper")
    check(not ok, "a helper is held to the helper budget, not main's")
    ok, _ = budget.fits(49152, "helper")
    check(ok, "and exactly the helper budget fits")


def test_what_if_includes_the_live_pool():
    """The default table used to list every candidate EXCEPT the one running."""
    table = budget.what_if()
    rows = [ln.split() for ln in table.splitlines()[1:]]
    pools = [int(r[0]) for r in rows]
    check(LIVE_POOL in pools, "what_if() has a row for 163840", str(pools))
    check(pools == sorted(pools), "rows are in ascending pool order", str(pools))
    live = next((r for r in rows if int(r[0]) == LIVE_POOL), None)
    if live:
        check(live[2:5] == ["114688", "49152", "0"],
              "and that row shows the documented split", " ".join(live))
        check(live[5] == "yes", "and says it fits the 16.3 GB card", " ".join(live))
    big = next((r for r in rows if int(r[0]) == 262144), None)
    check(big is not None and big[5] == "NO",
          "262144 is reported as NOT fitting, as documented",
          " ".join(big) if big else "missing")


def test_the_cap_layout():
    """Operator, 2026-09-28: always run from VRAM. The main share is the
    tiered cache's VRAM line (the served kv_vram_cells, engine 0041, or
    YAMADORI_MAIN_CAP), every conversation is advertised all of it, and the
    child's window is its own size -- not taken from main. (The 2026-09-28
    rule: the lane unranked, slots.LANE_RANK off; layout v2's lane is
    test_the_lane_comes_out_of_the_line.)"""
    import slots
    old = (budget._LINE, budget.MAIN_CAP, budget.CHILD_TOKENS)
    old_lr = slots.LANE_RANK
    slots.LANE_RANK = False
    try:
        budget._LINE, budget.MAIN_CAP, budget.CHILD_TOKENS = 163840, 0, 65536
        b = budget.budgets(393216)
        check(b["layout"] == "cap" and b["main"] == 163840
              and b["helper"] == 65536 and b["window"] == 163840
              and b["reserve"] == 393216 - 163840 - 65536
              and b["cap_source"] == "llama-server /props kv_vram_cells",
              "a served line of 163,840 in a pool of 393,216 (2 x cap + "
              "child): main = the line, the child 65,536, one request's "
              "window = the line, the rest a second conversation's", str(b))
        budget.MAIN_CAP = 158720
        b = budget.budgets(393216)
        check(b["main"] == 158720 and b["cap_source"] == "YAMADORI_MAIN_CAP",
              "YAMADORI_MAIN_CAP (the deploy's measured value) wins over the "
              "served line", str(b))
        budget.MAIN_CAP = 0
        budget._LINE = 0
        b = budget.budgets(163840)
        check(b["layout"] == "split",
              "kv_vram_cells 0 (a pool that is all VRAM): the split", str(b))
        budget._LINE = 400000
        b = budget.budgets(393216)
        check(b["layout"] == "split",
              "a line at or past the pool is no cap (nothing is tiered)",
              str(b))
        fake_server({f"{budget.DIRECT}/props": {
            "default_generation_settings": {"n_ctx": 393216},
            "total_slots": 3, "kv_vram_cells": 163840}})
        budget._LINE = None
        try:
            budget.pool_size(refresh=True)
            check(budget._LINE == 163840 and budget._SLOTS == 3,
                  "pool_size reads kv_vram_cells and total_slots from /props",
                  str((budget._LINE, budget._SLOTS)))
        finally:
            restore()
        import catalog
        import tiers
        budget._POOL, budget._LINE = 393216, 163840
        check(catalog.context_window() == 163840,
              "the advertised context_length is the main cap",
              str(catalog.context_window()))
        sh = tiers._shares()
        check(sh == {"main": 163840, "helper": 65536, "pool": 163840,
                     "capped": True},
              "tiers' shares: main, the child, one request's window (the cap)",
              str(sh))
        c = tiers.compaction_budget(12000, 120000, helper_active=1)
        check(c["window"] == 163840 and c["fits"],
              "a compaction's window is the cap, a running child or not (it "
              "swaps in while its conversation's main pauses)", str(c))
    finally:
        budget._LINE, budget.MAIN_CAP, budget.CHILD_TOKENS = old
        budget._POOL = None
        slots.LANE_RANK = old_lr


def test_the_lane_comes_out_of_the_line():
    """LAYOUT V2 (operator, 2026-09-29): "decider was the only thing that
    needed room". The lane (LANE_TOKENS, ranked above the primary so it is
    kept in VRAM) comes off the served line; YAMADORI_MAIN_CAP -- which the
    deploy writes as N - LANE -- is taken as it is."""
    import catalog
    import slots
    old = (budget._LINE, budget.MAIN_CAP, budget._POOL, slots.LANE_RANK)
    try:
        slots.LANE_RANK = True
        check(budget.LANE_TOKENS == 3072,
              "the lane is 3,072 cells: STATE_TOKENS 2,048 + the decider's "
              "measured non-state maximum (999) rounded up to 256-cell blocks",
              str(budget.LANE_TOKENS))
        budget._LINE, budget.MAIN_CAP = 166400, 0
        b = budget.budgets(262144)
        check(b["main"] == 166400 - 3072 and b["lane"] == 3072
              and b["window"] == 166400 - 3072
              and "less the lane" in b["cap_source"],
              "a served line of 166,400: main = the line less the lane",
              str(b))
        budget._POOL = 262144
        check(catalog.context_window() == 166400 - 3072,
              "the advertised context_length is the line less the lane",
              str(catalog.context_window()))
        budget.MAIN_CAP = 163328
        b = budget.budgets(262144)
        check(b["main"] == 163328 and b["cap_source"] == "YAMADORI_MAIN_CAP",
              "YAMADORI_MAIN_CAP (already N - LANE) is not reduced again",
              str(b))
        budget.MAIN_CAP = 0
        slots.LANE_RANK = False
        check(budget.budgets(262144)["main"] == 166400,
              "the lane unranked (slots.LANE_RANK off): not reserved; main is "
              "the line")
    finally:
        budget._LINE, budget.MAIN_CAP, budget._POOL, slots.LANE_RANK = old


def main() -> int:
    for fn in (test_the_fixture_cannot_reach_a_server,
               test_the_shipped_split_is_the_documented_one,
               test_the_invariants_hold_for_any_shares,
               test_the_floor_protects_the_conversation,
               test_pool_size_reads_the_server_and_caches,
               test_cap_for_answers_only_main_or_helper,
               test_fits_is_inclusive_and_explains_a_refusal,
               test_what_if_includes_the_live_pool,
               test_the_cap_layout,
               test_the_lane_comes_out_of_the_line):
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
