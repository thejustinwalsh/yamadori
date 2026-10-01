#!/usr/bin/env python
"""Where a request's wall clock goes: x_yamadori.timing {stage: ms}.

WHY (deploy check 2026-09-29): a side call took 72.8 s wall for a 414 ms generation, and nothing the proxy
recorded could say where the other ~72 s went -- no per-stage time existed.

HOW
  install()   (server.py, the proxy process only, at startup) wraps the functions a turn spends its time in -- the
              tier-model wait, the decider's prime, the session, prepare and what prepare calls (the ledger restore,
              the MCP offer, skills, library help), the window check, the warm wait, the A4000 room, the slot
              acquire / idle clear / release, the upstream generation, the records after it. A wrapper only times
              and never changes a call; a name a module no longer has is skipped (the removal of 2026-09-29 may
              delete some).
  begin()     (proxy._run_turn, first line) starts THIS request's timer on its cancel token -- the token job threads
              re-bind, so the second brain's and the decider's calls made for this request are counted too.
  record()    {wall_ms, stages {name: ms} (top-level calls only, so they do not overlap), nested {name: ms} (calls
              made inside a timed stage: detail, not summed), calls {name: n}, points {upstream_first_event_ms,
              upstream_prompt_ms, upstream_queue_est_ms}, accounted_ms, unaccounted_ms}.
  log_if_slow(rec)  one proxy.log line when the unaccounted time passes LOG_UNACCOUNTED_S.

THE UPSTREAM QUEUE. A request that llama-server holds behind another on the same slot shows as a first event long
after it was sent while the server's own prompt time (`timings.prompt_ms`) is short: `upstream_queue_est_ms` = the
first event less prompt_ms (an ESTIMATE: it also carries the first token's own decode and the network).

Nothing here is on when install() was not called: every function answers {} / does nothing.
"""
from __future__ import annotations

import functools
import importlib
import os
import threading
import time

import cancel

ATTR = "yamadori_timing"
# The coordinator's "a few seconds" (2026-09-29) for the log line: a LOGGING threshold only -- it changes no
# behaviour and no record. YAMADORI_TIMING_LOG_S moves it.
LOG_UNACCOUNTED_S = float(os.environ.get("YAMADORI_TIMING_LOG_S", "5") or 5)

# (module, function, kind): "call" a plain function, "gen" a generator (the upstream stream)
TARGETS = [
    ("max_mode", "wait_ready", "call"),
    ("decide_turn", "prime_for", "call"),
    ("proxy", "resolve_repo", "call"),
    ("proxy", "session_context", "call"),
    ("proxy", "prepare", "call"),
    ("proxy", "ledger_restore", "call"),
    ("proxy", "_mcp_offer", "call"),
    ("proxy", "_skills_tail", "call"),
    ("proxy", "_library_definitions", "call"),
    ("proxy", "_library_use", "call"),
    ("proxy", "check_client_prompt", "call"),
    ("proxy", "wait_for_warm", "call"),
    ("proxy", "_deep_record", "call"),
    ("proxy", "_x_yamadori", "call"),
    ("deep", "decide", "call"),
    ("decide_turn", "build_intent", "call"),
    ("decide_turn", "judge_stop", "call"),
    ("gpu_room", "ensure_room", "call"),
    ("slots", "acquire", "call"),
    ("slots", "clear_idle", "call"),
    ("slots", "release_idle", "call"),
    ("slots", "lane_kept_note", "call"),
    ("compaction", "record", "call"),
    ("token_ledger", "record_upstream", "call"),
    ("corpus", "log_turn", "call"),
    ("proxy", "_post_events_raw", "gen"),
]

_local = threading.local()
_installed: dict[str, bool] = {}


# ------------------------------------------------------------------ timer --
def _timer() -> dict | None:
    tok = cancel.current()
    return getattr(tok, ATTR, None) if tok is not None else None


def begin() -> dict | None:
    """Start the timer of the request this thread works for (its cancel token). None outside a request."""
    tok = cancel.current()
    if tok is None:
        return None
    t = {"t0": time.perf_counter(), "stages": {}, "nested": {}, "calls": {}, "points": {},
         "lock": threading.Lock()}
    setattr(tok, ATTR, t)
    return t


def _depth() -> int:
    return getattr(_local, "depth", 0)


def add(name: str, ms: float, nested: bool = False) -> None:
    t = _timer()
    if t is None:
        return
    with t["lock"]:
        bucket = t["nested" if nested else "stages"]
        bucket[name] = bucket.get(name, 0.0) + ms
        t["calls"][name] = t["calls"].get(name, 0) + 1


def point(name: str, value: float) -> None:
    t = _timer()
    if t is not None:
        with t["lock"]:
            t["points"].setdefault(name, round(value, 1))


def record() -> dict:
    """The request's timing, for x_yamadori.timing ({} outside a timed request)."""
    t = _timer()
    if t is None:
        return {}
    with t["lock"]:
        wall = (time.perf_counter() - t["t0"]) * 1000
        stages = {k: round(v, 1) for k, v in sorted(t["stages"].items(), key=lambda kv: -kv[1])}
        acc = sum(t["stages"].values())
        pts = dict(t["points"])
        return {"wall_ms": round(wall, 1), "stages": stages,
                "nested": {k: round(v, 1) for k, v in sorted(t["nested"].items(), key=lambda kv: -kv[1])},
                "calls": dict(t["calls"]), "points": pts,
                "accounted_ms": round(acc, 1), "unaccounted_ms": round(max(wall - acc, 0.0), 1)}


def log_if_slow(rec: dict, what: str = "request") -> str | None:
    """One proxy.log line when the unaccounted time passes LOG_UNACCOUNTED_S; the line, or None."""
    if not rec or rec.get("unaccounted_ms", 0) < LOG_UNACCOUNTED_S * 1000:
        return None
    top = ", ".join(f"{k} {v / 1000:.1f}s" for k, v in list(rec.get("stages", {}).items())[:5])
    pts = rec.get("points") or {}
    line = (f"  timing: {what} {rec['wall_ms'] / 1000:.1f}s wall, {rec['unaccounted_ms'] / 1000:.1f}s unaccounted"
            f" (over {LOG_UNACCOUNTED_S:g}s); stages: {top or 'none'}"
            + (f"; upstream first event {pts['upstream_first_event_ms'] / 1000:.1f}s, prompt "
               f"{(pts.get('upstream_prompt_ms') or 0) / 1000:.1f}s" if "upstream_first_event_ms" in pts else ""))
    print(line, flush=True)
    return line


# --------------------------------------------------------------- wrappers --
def _wrap_call(fn, name: str):
    @functools.wraps(fn)
    def timed(*a, **k):
        if _timer() is None:
            return fn(*a, **k)
        d = _depth()
        _local.depth = d + 1
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _local.depth = d
            add(name, (time.perf_counter() - t0) * 1000, nested=d > 0)
    timed.__yamadori_timed__ = True
    return timed


def _wrap_gen(fn, name: str):
    """A generator (the upstream stream): the time to its first event and to its end. Close and exceptions pass
    through unchanged."""
    @functools.wraps(fn)
    def timed(*a, **k):
        if _timer() is None:
            yield from fn(*a, **k)
            return
        nested = _depth() > 0
        t0 = time.perf_counter()
        first = True
        g = fn(*a, **k)
        try:
            while True:
                try:
                    v = next(g)
                except StopIteration as e:
                    return e.value
                if first:
                    first = False
                    point("upstream_first_event_ms", (time.perf_counter() - t0) * 1000)
                if isinstance(v, tuple) and len(v) == 2 and v[0] == "done" and isinstance(v[1], dict):
                    tm = v[1].get("_timings") or v[1].get("timings") or {}
                    pm = tm.get("prompt_ms") if isinstance(tm, dict) else None
                    if isinstance(pm, (int, float)):
                        point("upstream_prompt_ms", pm)
                        t = _timer()
                        fe = (t or {}).get("points", {}).get("upstream_first_event_ms")
                        if fe is not None:
                            point("upstream_queue_est_ms", max(fe - pm, 0.0))
                try:
                    yield v
                except GeneratorExit:
                    g.close()
                    raise
        finally:
            add(name, (time.perf_counter() - t0) * 1000, nested=nested)
    timed.__yamadori_timed__ = True
    return timed


def install(targets=None, modules: dict | None = None) -> dict:
    """Wrap every target that exists (idempotent). `modules` maps a name to a module object (tests). Returns
    {module.function: "wrapped" | "missing" | "already"}."""
    out = {}
    for mod_name, fn_name, kind in targets or TARGETS:
        key = f"{mod_name}.{fn_name}"
        try:
            mod = (modules or {}).get(mod_name) or importlib.import_module(mod_name)
        except Exception:                                            # noqa: BLE001
            out[key] = "missing"
            continue
        fn = getattr(mod, fn_name, None)
        if fn is None:
            out[key] = "missing"
            continue
        if getattr(fn, "__yamadori_timed__", False):
            out[key] = "already"
            continue
        label = f"{'upstream' if kind == 'gen' else mod_name}.{fn_name if kind != 'gen' else 'generate'}"
        setattr(mod, fn_name, (_wrap_gen if kind == "gen" else _wrap_call)(fn, label))
        out[key] = "wrapped"
        _installed[key] = True
    return out
