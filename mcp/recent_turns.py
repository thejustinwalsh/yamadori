#!/usr/bin/env python
"""The last few requests' decisions, in memory, for the tokonoma.

The proxy already builds one record per request, `x_yamadori` (proxy.
_x_yamadori), and hands it to the client. Nothing kept it. note() keeps the
two parts the tree draws -- what fan-out did and what recall injected --
from the most recent MAX_TURNS requests. No text, no account, no disk: it
lives in the proxy process and is gone on restart, which the tree shows as
INERT until the next request.

Called from exactly one place, after `d["x_yamadori"] = _x_yamadori(...)`
in proxy._run_turn. Never raises.
"""
from __future__ import annotations

import threading
import time
from collections import deque

MAX_TURNS = 50
# Fan-out: the most recent fan-out stays drawn for FANOUT_HOLD_S after its
# request, then retracts to arity 1 over FANOUT_DECAY_S. Chosen for the
# eye (a request takes 30-200 s, so a fork is still up while its answer is
# read), not measured.
FANOUT_HOLD_S = 120.0
FANOUT_DECAY_S = 60.0
# Foliage: recall injections over the task turns (utility calls excluded)
# of the last FOLIAGE_WINDOW_S, at most FOLIAGE_TURNS of them. A mean of
# FOLIAGE_FULL injected items per turn is full foliage. Choices, not
# measurements.
FOLIAGE_WINDOW_S = 1800.0
FOLIAGE_TURNS = 20
FOLIAGE_FULL = 3.0

_lock = threading.Lock()
_turns: deque[dict] = deque(maxlen=MAX_TURNS)


def _fanout_of(fan) -> dict | None:
    """arity, chosen and culled from an x_yamadori.fanout record; None when
    this request did not fan out (or fan-out failed before a candidate)."""
    if not isinstance(fan, dict) or fan.get("error"):
        return None
    cands = [c for c in fan.get("candidates") or [] if isinstance(c, dict)]
    arity = len(cands) or int(fan.get("n") or 0)
    if arity < 2:
        return None
    w = fan.get("winner_index")
    chosen = int(w) if isinstance(w, int) and 0 <= w < arity else 0
    return {"arity": arity, "chosen": chosen,
            "culled": [i for i in range(arity) if i != chosen],
            "selection": fan.get("selection")}


def _recall_of(x: dict) -> dict:
    """What recall put in this request: the path (YAMADORI_RECALL), how many
    items, and their tokens where the path reports them."""
    sk = x.get("skills") if isinstance(x.get("skills"), dict) else {}
    path = sk.get("path") or "hints"
    if path == "skills":
        ids = sk.get("ids") or []
        tok = sk.get("tokens")
        return {"path": "skills", "count": len(ids) if isinstance(ids, list) else 0,
                "tokens": int(tok) if isinstance(tok, (int, float)) else None}
    hints = x.get("hints") if isinstance(x.get("hints"), list) else []
    return {"path": "hints", "count": len(hints), "tokens": None}


def note(x: dict | None, now: float | None = None) -> None:
    try:
        if not isinstance(x, dict):
            return
        rec = {"at": time.time() if now is None else float(now),
               "utility": bool(x.get("utility")),
               "fanout": _fanout_of(x.get("fanout")),
               "recall": _recall_of(x)}
        with _lock:
            _turns.append(rec)
    except Exception:                                            # noqa: BLE001
        pass


def reset() -> None:
    with _lock:
        _turns.clear()


def summary(now: float | None = None) -> dict:
    """{turns, fanout, foliage} for vitals.snapshot()['tree']."""
    now = time.time() if now is None else now
    with _lock:
        turns = list(_turns)
    fan = None
    for t in reversed(turns):
        if t["fanout"] and not t["utility"]:
            age = max(0.0, now - t["at"])
            if age <= FANOUT_HOLD_S + FANOUT_DECAY_S:
                fade = 1.0 if age <= FANOUT_HOLD_S else \
                    max(0.0, 1 - (age - FANOUT_HOLD_S) / FANOUT_DECAY_S)
                fan = dict(t["fanout"], age_s=round(age, 1), fade=round(fade, 3))
            break
    recent = [t for t in turns if not t["utility"] and now - t["at"] <= FOLIAGE_WINDOW_S]
    recent = recent[-FOLIAGE_TURNS:]
    foliage = None
    if recent:
        counts = [t["recall"]["count"] for t in recent]
        toks = [t["recall"]["tokens"] for t in recent if t["recall"]["tokens"] is not None]
        paths = sorted({t["recall"]["path"] for t in recent})
        mean = sum(counts) / len(counts)
        foliage = {"turns": len(recent), "items": sum(counts),
                   "tokens": sum(toks) if toks else None,
                   "mean_per_turn": round(mean, 3),
                   "density": round(min(1.0, mean / FOLIAGE_FULL), 3),
                   "path": paths[0] if len(paths) == 1 else "+".join(paths)}
    return {"turns": len(turns), "fanout": fan, "foliage": foliage,
            "hold_s": FANOUT_HOLD_S, "decay_s": FANOUT_DECAY_S,
            "foliage_window_s": FOLIAGE_WINDOW_S, "foliage_full": FOLIAGE_FULL}
