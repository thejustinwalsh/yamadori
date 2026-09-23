"""How many requests may be in flight, derived from the KV pool.

WHY A CAP AT ALL

Three benchmark processes ran against this proxy at once. Nobody decided that;
each was started with `nohup` without stopping the last, and the proxy
accepted all of them. They did not fail cleanly -- they degraded each other
into 502s that took an hour to attribute, and the arm that happened to lose
most looked like the arm with a bug.

A server that accepts more work than it can do does not go faster. It goes
wrong in a way that looks like a different problem.

WHERE THE NUMBER COMES FROM

Not a guess. llama-server holds ONE unified KV pool -- measured at 147,456
tokens, since four slots reporting 131,072 each would be 22 GB of KV against
a measured 5.8 GB. `budget.py` splits it (the operator's split of
2026-09-22): 5/8 for the conversation and 3/8 for ONE helper -- main 92,160
and helper 55,296 at that pool (102,400 + 61,440 at the 163,840 config.yaml
launches with now).

The main budget is 5/8 of the pool, so two main requests at full budget would
want 10/8 of it -- more than exists, before the helper takes anything. The
honest ceiling with the helper running is ONE main request at full budget plus
one helper. Everything above that is oversubscription that works only because
most requests are nowhere near their ceiling.

(Earlier the same day this read "main 73,728, 36,864 for EACH of up to two
helpers": the 1/2 + 2 x 1/4 split, retired because two concurrent helpers
needed two conversations investigating at once -- mcp/budget.py. Before that,
"main 88,473, helper 36,864" and one helper lane: the 60/25/15 split, which
had no measurement behind it -- docs/CONSTRAINTS.md item 19.)

OVERSUBSCRIPTION IS ALLOWED, AND STATED

`MAIN_LANES` defaults to 2 rather than 1 because a typical turn is a few
thousand tokens, not 92,000, and a server that serialises everything to guard
against the worst case is slow all the time to prevent a rare failure. The
multiplier is explicit so that when it does bite, the cause is one number
someone can find rather than emergent behaviour.

WHAT HAPPENS WHEN FULL

Queue briefly, then refuse with 429 and Retry-After. Not queue indefinitely:
an unbounded queue converts overload into latency, the caller times out
anyway, and the work is done for nobody. A refusal is information; a timeout
is not.

Helpers get their own lane. A second hemisphere investigation must never
take the slot its own conversation is waiting on -- that is a deadlock, not
contention.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time

MAIN_LANES = int(os.environ.get("YAMADORI_MAIN_LANES", "2"))
# ONE second brain at a time, with 3/8 of the pool (see mcp/budget.py). It
# was briefly 2 on 2026-09-22, a quarter of the pool each; a second concurrent
# investigation now waits for this lane and is refused if it cannot get it.
HELPER_LANES = int(os.environ.get("YAMADORI_HELPER_LANES", "1"))
# How long a request waits for a lane before being refused. Long enough to
# ride out a burst, short enough that the caller is not left guessing.
WAIT_SECONDS = float(os.environ.get("YAMADORI_ADMIT_WAIT", "20"))

_lanes: dict[str, asyncio.Semaphore] = {}
_stats = {"admitted": 0, "queued": 0, "refused": 0, "peak_main": 0}
_inflight = {"main": 0, "helper": 0}


def _sem(kind: str) -> asyncio.Semaphore:
    if kind not in _lanes:
        _lanes[kind] = asyncio.Semaphore(
            MAIN_LANES if kind == "main" else HELPER_LANES)
    return _lanes[kind]


class Full(Exception):
    """No lane became free in time. Carries what the caller should do."""

    def __init__(self, kind: str, waited: float):
        self.kind = kind
        self.waited = waited
        lanes = MAIN_LANES if kind == "main" else HELPER_LANES
        super().__init__(
            f"all {lanes} {kind} lanes busy after {waited:.0f}s. This server "
            f"runs one model over a shared KV pool; more concurrent work does "
            f"not go faster, it goes wrong. Retry shortly.")


class admit:
    """Async context manager holding one lane for the life of a request."""

    def __init__(self, kind: str = "main"):
        self.kind = kind if kind in ("main", "helper") else "main"
        self._held = False

    async def __aenter__(self):
        sem = _sem(self.kind)
        t0 = time.time()
        if sem.locked():
            _stats["queued"] += 1
        try:
            await asyncio.wait_for(sem.acquire(), timeout=WAIT_SECONDS)
        except asyncio.TimeoutError:
            _stats["refused"] += 1
            raise Full(self.kind, time.time() - t0) from None
        self._held = True
        _stats["admitted"] += 1
        _inflight[self.kind] += 1
        _stats["peak_main"] = max(_stats["peak_main"], _inflight["main"])
        return self

    async def __aexit__(self, *exc):
        if self._held:
            _inflight[self.kind] -= 1
            _sem(self.kind).release()
            self._held = False
        return False


# ---------------------------------------------------------------------------
# AT MOST HELPER_LANES SECOND BRAINS, AND THIS IS WHERE THAT BECOMES TRUE.
#
# The lane count used to be only a statement: `shomen.investigate()` runs
# inside a worker thread and calls the model directly, never touching the
# lane above. So the declared topology held only while exactly one request was
# in flight. (HELPER_LANES was briefly 2 on 2026-09-22, a quarter of the pool
# each; it is 1 again, with 3/8 of the pool, because two at once needed two
# conversations investigating simultaneously -- mcp/budget.py. A second
# concurrent investigation waits here and is refused.)
#
# It needs a THREADING semaphore rather than the asyncio one. The lanes above
# are acquired on the event loop; the hemisphere is not on the loop, and
# `asyncio.Semaphore` from a worker thread either does nothing useful or
# raises. Two mechanisms because there are genuinely two contexts, not because
# one of them is wrong.
#
# Blocking rather than refusing: a second brain that waits its turn delays one
# answer, while a second brain that runs anyway puts two extra contexts on a
# card with 3.9 GB free, which is the condition that killed a benchmark run.
# ---------------------------------------------------------------------------
_helper_sem = threading.Semaphore(HELPER_LANES)
_helper_stats = {"granted": 0, "waited": 0, "timed_out": 0, "inflight": 0}


class helper_lane:
    """Hold a helper lane (one of HELPER_LANES) for the life of one investigation.

    Synchronous, so it can be used from the tool loop. Returns False from
    __enter__ if the lane never came free, which the caller reports rather
    than ignoring -- an investigation that silently did not run is worse than
    one that says it was too busy.
    """

    def __init__(self, timeout: float = WAIT_SECONDS):
        self.timeout = timeout
        self.held = False

    def __enter__(self) -> bool:
        t0 = time.time()
        if not _helper_sem.acquire(blocking=False):
            _helper_stats["waited"] += 1
            self.held = _helper_sem.acquire(timeout=self.timeout)
        else:
            self.held = True
        if self.held:
            _helper_stats["granted"] += 1
            _helper_stats["inflight"] += 1
        else:
            _helper_stats["timed_out"] += 1
            print(f"  helper lane busy after {time.time() - t0:.0f}s; "
                  f"investigation skipped", flush=True)
        return self.held

    def __exit__(self, *exc) -> bool:
        if self.held:
            _helper_stats["inflight"] -= 1
            _helper_sem.release()
            self.held = False
        return False


# ---------------------------------------------------------------------------
# THE IMAGE LANE. Image generation is a GPU consumer on CUDA1, a different card
# from the chat model, so it takes neither a main lane nor the helper lane: an
# image that takes a minute must never make a chat request wait, and a chat
# request must never make an image wait. It gets its own lane, ONE, because two
# denoisers on one 16 GB card beside the resident retrieval models is the
# "two consumers on one card" failure AGENTS.md names -- the loser looks like
# the one with the bug.
#
# Threading, not asyncio, for the helper lane's reason: the model's
# generate_image tool runs inside the tool loop's worker thread, and the
# /v1/images/generations route runs its call in the threadpool. Both paths
# take this one semaphore, so the count holds across them.
# ---------------------------------------------------------------------------
IMAGE_LANES = int(os.environ.get("YAMADORI_IMAGE_LANES", "1"))
IMAGE_WAIT_SECONDS = float(os.environ.get("YAMADORI_IMAGE_WAIT", "30"))
_image_sem = threading.Semaphore(IMAGE_LANES)
_image_stats = {"granted": 0, "waited": 0, "timed_out": 0, "inflight": 0}


class image_lane:
    """Hold the image lane for one generation. __enter__ returns False if it
    never came free, which the caller reports as busy -- never as a failure."""

    def __init__(self, timeout: float | None = None):
        self.timeout = IMAGE_WAIT_SECONDS if timeout is None else timeout
        self.held = False

    def __enter__(self) -> bool:
        if not _image_sem.acquire(blocking=False):
            _image_stats["waited"] += 1
            self.held = _image_sem.acquire(timeout=self.timeout)
        else:
            self.held = True
        if self.held:
            _image_stats["granted"] += 1
            _image_stats["inflight"] += 1
        else:
            _image_stats["timed_out"] += 1
        return self.held

    def __exit__(self, *exc) -> bool:
        if self.held:
            _image_stats["inflight"] -= 1
            _image_sem.release()
            self.held = False
        return False


def snapshot() -> dict:
    """For the vitals page: what the cap is and whether it is biting."""
    try:
        import budget
        b = budget.budgets()
    except Exception:                                            # noqa: BLE001
        b = {}
    return {
        "main_lanes": MAIN_LANES, "helper_lanes": HELPER_LANES,
        "inflight": dict(_inflight), "wait_seconds": WAIT_SECONDS,
        "helper": dict(_helper_stats),
        "image_lanes": IMAGE_LANES, "image": dict(_image_stats),
        **_stats,
        "pool": b.get("pool"), "main_budget": b.get("main"),
        # The honest ceiling if every request used its full budget. When
        # MAIN_LANES exceeds this, the server is oversubscribed on purpose.
        "full_budget_ceiling": (b["pool"] // b["main"]) if b.get("main") else None,
    }
