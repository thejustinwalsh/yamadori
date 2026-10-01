#!/usr/bin/env python
"""IS THE STACK IDLE? The one definition every idle-gated job reads.

    python mcp/idle.py            print the verdict now

Operator, 2026-09-27 (docs/PACKAGE-ONBOARDING.md section 3, decisions 5 and
6): the gpu stages of a package onboarding (`package.index`, `package.knn`,
the evaluation's model-decider arm), the skill model stages of an
onboarding's skills and the two index rebuilds (`skill.match_index`,
`package.example_knn_index`) run only when the stack is idle -- one GPU
consumer at a time -- and "IDLE_MINUTES: don't take 15 as given. Use the
existing idle definition (skill_learn.idle_state plus /slots) and mark the
number with its source".

THE DEFINITION (the parts of skill_learn.idle_state that are about the
stack, plus the check it did not have):

  1. REQUESTS. No client request for `skill_learn.IDLE_MINUTES` minutes: the
     newest `turn` event in the corpus database, which the proxy writes for
     every request (`last_request_at`, moved here from skill_learn, which now
     calls it). A corpus that cannot be read is NOT idle. THE NUMBER'S
     SOURCE: skill_learn.IDLE_MINUTES (15, env YAMADORI_SKILL_LEARN_IDLE_
     MINUTES), the learner's existing value: operator, 2026-09-27, "keep
     ours (no external evidence)" (`IDLE_MINUTES_SOURCE`); onboarding adds
     no number of its own and moves with it.
  2. THE CARD. No OTHER gpu-lane job running (the lane runs one at a time, so
     this is a job that started outside the lane's accounting, e.g. before a
     restart).
  3. THE MODEL. No llama-server slot of the main model is generating
     (`/slots` `is_processing`, which the watchdog, `slots` and
     `model.release_slot` already read) -- so a long generation that started
     before the quiet window is not overrun. Read only when llama-swap's
     `/running` (which never loads anything) lists the main model: asking a
     model's /slots through llama-swap LOADS it when it is unloaded
     (docs/ENGINES.md, "The PTQ1_0 PDL race"). /running unreadable is NOT
     idle (nothing can be said about the card).

`until` is DERIVED, never a number of ours: the last request's time plus
IDLE_MINUTES when the requests decide it; otherwise the worker's next
scheduler tick (worker.RECLAIM_EVERY), when a slot is generating or another
gpu job is running.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MAIN_MODEL = os.environ.get("YAMADORI_MODEL", "bonsai")

IDLE_MINUTES_SOURCE = (
    "skill_learn.IDLE_MINUTES (env YAMADORI_SKILL_LEARN_IDLE_MINUTES), the "
    "skill learner's existing idle window: operator, 2026-09-27: keep ours "
    "(no external evidence)")


def idle_minutes() -> float:
    import skill_learn
    return float(skill_learn.IDLE_MINUTES)


def last_request_at() -> tuple[float | None, str]:
    """(time of the newest client request, how it was read). None when the
    corpus cannot be read -- which is not evidence of idleness."""
    try:
        import corpus
        path = os.path.abspath(corpus.CORPUS_DB)
    except Exception as e:                                       # noqa: BLE001
        return None, f"the corpus module did not load ({e})"
    if not os.path.exists(path):
        return None, "no corpus database, so no evidence of idleness"
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            ts = con.execute("SELECT MAX(ts) FROM events WHERE kind='turn'"
                             ).fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as e:
        return None, f"the corpus could not be read ({e})"
    return (float(ts) if ts else 0.0), "read"


def next_tick() -> float:
    """When the worker next looks at its schedules (worker.RECLAIM_EVERY)."""
    try:
        import worker
        every = float(worker.RECLAIM_EVERY)
    except Exception:                                            # noqa: BLE001
        every = 60.0          # worker.RECLAIM_EVERY's value, when not importable
    return time.time() + every


def slots_processing(model: str | None = None) -> tuple[list[int] | None, str]:
    """(the main model's -- or `model`'s -- slots that are generating, how it
    was read). [] when none is, or when the model is not loaded; None when it
    cannot be told. A read only while /running lists it (never loads)."""
    MAIN_MODEL_ = model or MAIN_MODEL
    try:
        import gpu_room
        rows = gpu_room.running(UPSTREAM)
    except Exception as e:                                       # noqa: BLE001
        return None, f"llama-swap /running: {type(e).__name__}: {e}"[:200]
    if rows is None:
        return None, "llama-swap /running could not be read"
    if not any(str(r.get("model")) == MAIN_MODEL_ for r in rows):
        return [], f"{MAIN_MODEL_} is not loaded"
    try:
        with urllib.request.urlopen(
                f"{UPSTREAM}/upstream/{MAIN_MODEL_}/slots", timeout=10) as r:
            table = json.loads(r.read().decode("utf-8") or "[]")
    except Exception as e:                                       # noqa: BLE001
        return None, f"/slots: {type(e).__name__}: {e}"[:200]
    if not isinstance(table, list):
        return None, "/slots did not answer a list"
    return [s.get("id") for s in table if isinstance(s, dict)
            and s.get("is_processing")], "read"


def gpu_running(exclude: str | None = None, scope: str | None = None) -> int:
    """Other gpu jobs running: all of them (as before), or those of one gpu SCOPE (mcp/jobs.py GPU_SCOPES)."""
    import jobs
    con = jobs._db()
    try:
        if scope in jobs.GPU_SCOPES:
            return con.execute(
                "SELECT COUNT(*) FROM jobs WHERE lane IN ('gpu','gpu_a4000') AND state='running' "
                "AND COALESCE(card, lane)=? AND id != ?", (scope, exclude or "")).fetchone()[0]
        return con.execute(
            "SELECT COUNT(*) FROM jobs WHERE lane='gpu' AND state='running' "
            "AND id != ?", (exclude or "",)).fetchone()[0]
    finally:
        con.close()


# The A4000's generating server, for a job of its scope: the table's helper (bonsai-a4000), where the operator's
# own conversation's jjava and side calls run while the main card is locked.
A4000_MODEL = os.environ.get("YAMADORI_A4000_MODEL", "bonsai-a4000")


def stack_idle(job_id: str | None = None, scope: str | None = None) -> dict:
    """{idle, why, until, checks}. Every condition is checked in order; the
    first that fails is the reason, and `until` is when looking again can
    change the answer. `scope` gpu_a4000 (a job claimed for the A4000,
    mcp/jobs.py): the main card's state is not its business -- it waits for
    no client request, no other job of ITS scope, and the A4000's generating
    server idle; never a read of the main card."""
    checks: dict = {"idle_minutes": idle_minutes(),
                    "idle_minutes_source": IDLE_MINUTES_SOURCE, "scope": scope or "gpu"}
    import max_mode
    scope = scope or max_mode.scope()        # the worker's thread sets it (worker.run_one)
    checks["scope"] = scope or "gpu"
    a4000 = scope == "gpu_a4000"
    if not a4000 and max_mode.blocks(MAIN_MODEL):
        # MAX MODE (mcp/max_mode.py), first: the max model holds the card; "not loaded" is not idle, and a job that
        # asked the main model would load it
        checks["max_mode"] = max_mode.snapshot()
        return {"idle": False, "why": max_mode.blocked_reason(MAIN_MODEL), "until": next_tick(),
                "checks": checks}
    at, how = last_request_at()
    checks["last_request_at"] = at
    if at is None:
        return {"idle": False, "why": how, "until": next_tick(),
                "checks": checks}
    quiet = (time.time() - at) / 60 if at else float("inf")
    checks["quiet_minutes"] = round(quiet, 2) if at else None
    if quiet < checks["idle_minutes"]:
        return {"idle": False,
                "why": f"the last client request was {quiet:.1f} min ago "
                       f"(idle means {checks['idle_minutes']:g}: "
                       "skill_learn.IDLE_MINUTES)",
                "until": at + checks["idle_minutes"] * 60, "checks": checks}
    n = gpu_running(job_id, scope=scope if a4000 else None)
    checks["gpu_running"] = n
    if n:
        return {"idle": False, "why": f"{n} other gpu-lane job(s) running",
                "until": next_tick(), "checks": checks}
    busy, how = slots_processing(A4000_MODEL) if a4000 else slots_processing()
    checks["slots"] = {"processing": busy, "how": how}
    if busy is None:
        return {"idle": False, "why": f"cannot tell whether the model is "
                                      f"generating ({how})",
                "until": next_tick(), "checks": checks}
    if busy:
        return {"idle": False, "why": f"{A4000_MODEL if a4000 else MAIN_MODEL} slot(s) {busy} are "
                                      "generating",
                "until": next_tick(), "checks": checks}
    return {"idle": True,
            "why": (f"no request for {quiet:.0f} min" if at else
                    "no request on record")
            + f"; no other gpu job; {how if how != 'read' else 'no slot generating'}",
            "until": None, "checks": checks}


if __name__ == "__main__":
    print(json.dumps(stack_idle(), indent=1, default=str))
