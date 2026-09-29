#!/usr/bin/env python
"""One model per effort tier: which main model holds the 5060 Ti, and the swaps between them (docs/FLASH-NEXT.md
sections 4-5; mcp/tier_models.py holds the table). The module keeps its 2026-09-28 name, `max_mode`, because thirty
call sites import it; since 2026-09-29 it serves the whole tier -> model table, of which max mode is one row.

THE RULES (operator, 2026-09-27/28/29)

  - Each tier is served by the model the table names (mcp/tier_models.py; today's table: minimal..high -> bonsai,
    xhigh -> mirai-s, max -> flash-next). ONE big model is loaded on the 5060 Ti at a time: they swap each other
    inside llama-swap's `primary` group (swap: true).
  - EVERY call of a conversation goes to its model: main, the second brain, summaries, compaction, its side calls,
    the decider's questions, warms (the model is bound to the request's cancel token: set_current / current).
  - Nothing may make llama-swap load a main model while another holds the card, except a request admitted to it.
  - "If a medium tier comes in on max mode, we reject it. Plain and simple, model is at capacity error." (operator,
    2026-09-28). GENERALISED to the table and CONFIRMED (operator, 2026-09-29: "swap order is fine"): tiers are
    ordered, max > xhigh > the Bonsai tiers; a request for a
    LOWER tier's model while a HIGHER tier's model holds the card is refused at once -- HTTP 503 model_at_capacity
    with Retry-After (api_errors.at_capacity), recorded in x_yamadori.capacity; a request for a HIGHER tier's model
    while a lower one has work in flight WAITS for that work (never cancels it) and then takes the card. A waiter
    that a still higher tier overtakes before it starts is refused the same way (it never ran).
  - A client's side call (a title, a classifier, a flattened compaction: selection.utility_call) names no
    conversation: it is served by whichever main model holds or is loaded on the card, never refused.
  - The return: a lower tier's request is served once the higher one has gone idle -- no request of it in flight,
    and (only when the operator gives one) IDLE_S seconds since its last (YAMADORI_TIER_IDLE_S, or the older
    YAMADORI_MAX_IDLE_S). Unset, the swap back happens on the next lower request with nothing higher in flight.

THE SWAP ITSELF (wait_ready): once the other models' work has drained, the first request for a model that is not
loaded LOADS it explicitly (GET /upstream/<model>/health: llama-swap starts it and the group swap stops the others),
timed, and then reads /running to confirm that no other main model stayed loaded -- a reload leak is recorded, never
silent. x_yamadori.capacity.swap {from, to, load_s, left_loaded}.

WHERE IT IS DECIDED

  server._serve_turn calls decide() before admission (tier and utility read from the body exactly as prepare()
  will), refuses through api_errors, or takes a Lease (in-flight counts, released when the request ends) and puts the
  model in body["_upstream_model"]. proxy._run_turn binds it to the request's cancel token (set_current) and waits for
  the other models' work to drain (wait_ready); prepare() routes on it. Internal callers read current() -- the one
  door (model.shape) defaults to it, so the second brain, the decider and summaries inherit the conversation's model
  on every thread bound to the request's token.

  Callers with no conversation (the worker, the tools API, the decider's prime, deploy_check) never load a model that
  is not serving: blocks(model) is True for a main model while another holds the card (llama-swap's GET /running,
  which never loads anything, and the state file this module writes for the other processes).

OFF BY DEFAULT: with no table (YAMADORI_TIER_MODELS and YAMADORI_MAX_MODEL unset) every function answers exactly what
the stack did before (one model, nothing refused, nothing blocked).
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cancel  # noqa: E402
import tier_models  # noqa: E402

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
TABLE = tier_models.reload()
MAIN = TABLE.default
ENABLED = TABLE.enabled
MAX_TIER = "max"
# The max tier's model when it is not the default ("" otherwise): kept for the callers that name max mode.
MAX = TABLE.model_for(MAX_TIER) if ENABLED and TABLE.model_for(MAX_TIER) != MAIN else ""
MODELS = TABLE.main_models()
# The full table (a table FILE): the explicit timed swap and the blocking of every main model. Without it (the
# 2026-09-28 YAMADORI_MAX_MODEL) exactly the max-mode behaviour of that day: only the default model is ever blocked,
# and the first upstream call of the request loads its model.
FULL = TABLE.profiles_on


def _idle_s() -> float | None:
    v = (os.environ.get("YAMADORI_TIER_IDLE_S") or os.environ.get("YAMADORI_MAX_IDLE_S") or "").strip()
    try:
        return float(v) if v else None
    except ValueError:
        return None


# The operator's value, not ours (None until given): seconds a higher tier's model must be idle before a lower tier's
# request may swap it out.
IDLE_S = _idle_s()
# api_errors' existing Retry-After for a 503 the model server gives while loading (api_errors.of_upstream): used
# when the end of the work in flight cannot be known.
RETRY_AFTER_UNKNOWN = 30
RUNNING_TTL_S = float(os.environ.get("YAMADORI_MAX_RUNNING_TTL_S", "2"))
# llama-swap's own healthCheckTimeout (config.yaml: 900 s): the longest a load may take before llama-swap gives up.
LOAD_TIMEOUT_S = 900

_lock = threading.Condition()
_inflight: dict[str, int] = {}
_switching_to: str | None = None
_last_end: dict[str, float] = {}
_last_swap: dict = {}
_running_cache: tuple[float, set[str]] = (0.0, set())


# ------------------------------------------------------------------ models --
def rank(model: str | None) -> int:
    return TABLE.rank(model)


def is_main(model: str | None) -> bool:
    """One of the main models (the ones that hold the conversation slots)."""
    return bool(model) and (model == MAIN or (ENABLED and model in MODELS))


def model_for(tier_name: str | None, utility: bool = False) -> str:
    """The upstream model a request goes to (no state: see decide() for the state-dependent part)."""
    if not ENABLED:
        return MAIN
    if utility:
        on = card_model()
        if on:
            return on
        return MAIN
    return TABLE.model_for(tier_name)


def set_current(model: str | None) -> None:
    """Bind the conversation's model to this request (its cancel token, which job threads re-bind)."""
    tok = cancel.current()
    if tok is not None and model:
        setattr(tok, "yamadori_model", model)


def current(default: str | None = None) -> str:
    """The model of the request this thread works for; else `default` (else the main model)."""
    tok = cancel.current()
    m = getattr(tok, "yamadori_model", None) if tok is not None else None
    return m or default or MAIN


def bound_model() -> str | None:
    """The model bound to the request this thread works for, or None outside a request."""
    tok = cancel.current()
    return getattr(tok, "yamadori_model", None) if tok is not None else None


class ModelAtCapacity(RuntimeError):
    """A caller asked for a model that must not be loaded now (blocks()). Retryable once the holder frees the card."""

    def __init__(self, model: str, holder: str | None = None, why: str | None = None):
        self.holder = holder or holder_model() or MAX or "another model"
        super().__init__(why or blocked_reason(model))
        self.model = model
        self.retryable = True
        self.retry_after = retry_after()


def guard(model: str | None) -> None:
    """The one door's backstop (model.post): refuse a model that must not be loaded, unless this thread works for a
    request admitted to that very model (a swap the proxy decided)."""
    if model and blocks(model) and bound_model() != model:
        raise ModelAtCapacity(model)


def at_max() -> bool:
    """This thread works for a request a model other than the default serves (its card, not Bonsai's caps)."""
    return ENABLED and current() != MAIN


# ----------------------------------------------------------------- loaded --
def running() -> set[str]:
    """llama-swap's loaded models (GET /running reports; it never loads). Cached RUNNING_TTL_S."""
    global _running_cache
    t, ids = _running_cache
    if time.time() - t < RUNNING_TTL_S:
        return ids
    try:
        import gpu_room
        rows = gpu_room.running(UPSTREAM)
    except Exception:                                                # noqa: BLE001
        rows = None
    ids = {str(r.get("model")) for r in rows or [] if str(r.get("state", "ready")) != "stopped"}
    _running_cache = (time.time(), ids)
    return ids


def _fresh_running() -> set[str]:
    global _running_cache
    _running_cache = (0.0, set())
    return running()


def loaded(model: str) -> bool:
    return model in running()


def card_model() -> str | None:
    """The main model on the card now: the holder, else the one llama-swap reports loaded (highest rank first)."""
    h = holder_model()
    if h:
        return h
    on = [m for m in MODELS if loaded(m)]
    return max(on, key=rank) if on else None


# ONE PROCESS SWAPS THE CARD (operator's coordinator, 2026-09-29, after an unstubbed offline test of this module
# started flash-next on the running stack): only the proxy (server.py calls enable_swaps() at startup) or a process
# started with the explicit live flag YAMADORI_ALLOW_SWAP=1 (a bench script inside its GPU window) may make
# llama-swap load a model from here. Anywhere else _load refuses and says why; nothing is sent.
_SWAPS = {"on": os.environ.get("YAMADORI_ALLOW_SWAP") == "1"}


def enable_swaps(on: bool = True) -> None:
    _SWAPS["on"] = bool(on)


def _load(model: str, timeout: float = LOAD_TIMEOUT_S) -> tuple[bool, str]:
    """Make llama-swap start `model` (the group swap stops the others). The one deliberate /upstream/<model>/ read."""
    if os.environ.get("YAMADORI_OFFLINE_GUARD") == "1":
        # an offline suite (scripts/run_tests.py): never, whatever enable_swaps() said
        return False, "not sent: an offline suite (YAMADORI_OFFLINE_GUARD=1) never swaps the card"
    if not _SWAPS["on"]:
        return False, ("not sent: this process may not swap the card (only the proxy, or YAMADORI_ALLOW_SWAP=1); "
                       "the request's first upstream call loads the model")
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/upstream/{model}/health", timeout=timeout) as r:
            return r.status == 200, f"HTTP {r.status}"
    except Exception as e:                                           # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"[:200]


# ------------------------------------------------------------------ state --
def _state_path() -> str:
    p = os.environ.get("YAMADORI_MAX_STATE")
    if p:
        return p
    import gpu_room
    return os.path.join(gpu_room.state_dir(), "max_mode.json")


def _write_state() -> None:
    """For the other processes (worker, tools API): what this proxy is doing. Written only when a lease changes."""
    if not ENABLED:
        return
    rec = {"pid": os.getpid(), "t": time.time(), "switching_to": _switching_to,
           "inflight": dict(_inflight), "last_end": dict(_last_end),
           "last_max_end": _last_end.get(MAX, 0.0) if MAX else 0.0,
           "max": MAX, "main": MAIN, "models": MODELS}
    try:
        p = _state_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, p)
    except OSError:
        pass


def _read_state() -> dict:
    try:
        with open(_state_path(), encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    pid = d.get("pid")
    if pid and pid != os.getpid():
        try:
            import psutil
            if not psutil.pid_exists(int(pid)):
                return {}
        except Exception:                                            # noqa: BLE001
            pass
    return d if isinstance(d, dict) else {}


def _holder_of(inflight: dict, switching: str | None, last_end: dict) -> str | None:
    if switching:
        return switching
    busy = [m for m in MODELS if int(inflight.get(m, 0) or 0) > 0]
    if busy:
        return max(busy, key=rank)
    if IDLE_S is not None:
        now = time.time()
        recent = [m for m in MODELS if m != MAIN and float(last_end.get(m, 0) or 0) > 0
                  and now - float(last_end[m]) < IDLE_S and loaded(m)]
        if recent:
            return max(recent, key=rank)
    return None


def holder_model() -> str | None:
    """The main model that holds the card now: the one being switched to, else the highest-ranked with work in
    flight, else (with the operator's IDLE_S) a non-default model used less than IDLE_S ago and still loaded. In
    another process, from the proxy's state file. None: nobody holds it (whoever asks next may swap)."""
    if not ENABLED:
        return None
    with _lock:
        inflight, switching, last_end = dict(_inflight), _switching_to, dict(_last_end)
    mine = any(v > 0 for v in inflight.values())
    if not mine and not switching:
        st = _read_state()
        if st and st.get("pid") != os.getpid():         # this process's own state is the one in memory
            inflight = {k: int(v or 0) for k, v in (st.get("inflight") or {}).items()}
            switching = st.get("switching_to")
            last_end = {k: max(float(v or 0), last_end.get(k, 0.0))
                        for k, v in (st.get("last_end") or {}).items()} or last_end
    return _holder_of(inflight, switching, last_end)


def max_active() -> bool:
    """A model other than the default holds the card (the 2026-09-28 name: max mode is active)."""
    h = holder_model()
    return bool(h) and h != MAIN


def blocks(model: str | None) -> bool:
    """A caller must not touch `model` now: it is a main model and another holds the card, or another main model is
    loaded and this one is not. Any /upstream/<model>/... request would make llama-swap load it and swap the other
    out -- only a request admitted to `model` may do that (guard checks bound_model)."""
    if not ENABLED or not is_main(model):
        return False
    if not FULL and model != MAIN:
        return False                              # 2026-09-28: only the default model is ever blocked
    h = holder_model()
    if h and h != model:
        return True
    if loaded(model):
        return False
    return any(loaded(m) for m in MODELS if m != model)


def blocked_reason(model: str | None) -> str:
    h = holder_model() or card_model() or "another model"
    return (f"{model} is not served now: the card holds {h} (one model per effort tier, max mode); nothing may "
            f"load {model} until {h} frees it")


# --------------------------------------------------------------- requests --
class Decision:
    def __init__(self, model: str, tier: str | None, utility: bool, refuse: bool = False,
                 retry_after: int | None = None, why: str = "", holder: str | None = None):
        self.model, self.tier, self.utility = model, tier, utility
        self.refuse, self.retry_after, self.why, self.holder = refuse, retry_after, why, holder

    def record(self) -> dict:
        prof = {}
        try:
            p = TABLE.profile(self.model)
            prof = {"effort": (p.get("effort") or {}).get("value"),
                    "effort_class": (p.get("effort") or {}).get("class")}
        except Exception:                                            # noqa: BLE001
            pass
        return {"model": self.model, "tier": self.tier, "utility": self.utility, "refused": self.refuse,
                "retry_after": self.retry_after, "why": self.why, "enabled": ENABLED, "holder": self.holder,
                "rank": rank(self.model), "table": TABLE.source, **prof}


def requested_tier(body: dict) -> str | None:
    """The tier prepare() will resolve for this body (catalog hint, body effort, header)."""
    import catalog
    import tiers
    b = body
    _internal, hint, _known = catalog.resolve(body.get("model"))
    if hint and not body.get("reasoning_effort") and not (
            isinstance(body.get("reasoning"), dict) and body["reasoning"].get("effort")):
        b = dict(body, reasoning_effort=hint)
    try:
        return tiers.resolve(b, tiers.from_header(body.get("_features")))["name"]
    except Exception:                                                # noqa: BLE001
        return None


def is_utility(body: dict) -> bool:
    import proxy
    import system_roles
    msgs, _ = system_roles.one_system(body.get("messages") or [])
    try:
        return bool(proxy.utility_of(body, proxy.strip_thinking(msgs)).get("utility"))
    except Exception:                                                # noqa: BLE001
        return False


def decide(tier: str | None, utility: bool) -> Decision:
    """Where this request goes, or why it is refused. Pure given the state; no waiting."""
    if not ENABLED:
        return Decision(MAIN, tier, utility, why="one model (no tier table: YAMADORI_TIER_MODELS unset)")
    h = holder_model()
    if utility:
        on = h or card_model()
        if on:
            return Decision(on, tier, utility, holder=h,
                            why=f"a side call names no conversation: served by {on}, which is on the card")
        return Decision(MAIN, tier, utility, why="a side call with no main model on the card: the default model")
    want = TABLE.model_for(tier)
    why_tier = f"tier {tier} is served by {want} ({TABLE.source})"
    if h and h != want:
        if rank(want) > rank(h):
            return Decision(want, tier, utility, holder=h,
                            why=f"{why_tier}; it outranks {h}, which holds the card: waits for its work, then swaps")
        return Decision(want, tier, utility, refuse=True, retry_after=retry_after(), holder=h,
                        why=f"{why_tier}; {h} (a higher tier's model) holds the card")
    on = card_model()
    swap = f" (swap from {on})" if on and on != want else ""
    return Decision(want, tier, utility, holder=h, why=why_tier + swap)


def retry_after() -> int:
    """Seconds until a lower tier's request may be served: the rest of the operator's idle period when nothing is in
    flight; else unknown (api_errors' 503 value)."""
    with _lock:
        busy = any(v > 0 for v in _inflight.values()) or _switching_to is not None
        last = max(_last_end.values()) if _last_end else 0.0
    if not busy and IDLE_S is not None and last > 0:
        return max(1, int(math.ceil(IDLE_S - (time.time() - last))))
    return RETRY_AFTER_UNKNOWN


class Lease:
    """A request's hold on its model: counted from admission until it ends (release()). A lease for a model that
    outranks another model with work in flight marks the switch, so new lower work is refused from that moment."""

    def __init__(self, decision: Decision):
        global _switching_to
        self.model = decision.model
        self.decision = decision
        self.released = False
        self.t0 = time.time()
        with _lock:
            _inflight[self.model] = _inflight.get(self.model, 0) + 1
            others = [m for m, n in _inflight.items() if m != self.model and n > 0]
            if ENABLED and others and all(rank(self.model) > rank(m) for m in others) and \
                    (_switching_to is None or rank(self.model) > rank(_switching_to)):
                _switching_to = self.model
            _lock.notify_all()
        _write_state()

    def release(self) -> None:
        global _switching_to
        if self.released:
            return
        self.released = True
        with _lock:
            _inflight[self.model] = max(0, _inflight.get(self.model, 0) - 1)
            _last_end[self.model] = time.time()
            if _switching_to is not None and (
                    not any(n > 0 for n in _inflight.values())
                    or (FULL and _inflight.get(_switching_to, 0) == 0)):
                # nothing in flight, or (the full table) the model being switched to has no request left (its
                # waiter went away): nothing is switching. Without the table file: 2026-09-28's rule exactly.
                _switching_to = None
            _lock.notify_all()
        _write_state()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def release(lease: "Lease | None") -> None:
    if lease is not None:
        lease.release()


def wait_ready(model: str | None, poll_s: float = 0.5) -> dict:
    """Block until no request of ANOTHER main model is in flight (never cancelling it), then load `model` if it is
    not loaded (the swap), timed, and confirm no other main model stayed loaded. Honours the request's cancel; a
    waiter overtaken by a still higher tier is refused (ModelAtCapacity: it never ran). Returns
    {waited_s, other_inflight_at_start, swap?}."""
    global _switching_to, _last_swap
    out: dict = {"waited_s": 0.0, "other_inflight_at_start": 0}
    if not ENABLED or not is_main(model):
        return out
    t0 = time.time()
    with _lock:
        out["other_inflight_at_start"] = sum(n for m, n in _inflight.items() if m != model)
        while any(n > 0 for m, n in _inflight.items() if m != model):
            cancel.check()
            sw = _switching_to
            if sw and sw != model and rank(sw) > rank(model):
                raise ModelAtCapacity(model, holder=sw, why=(
                    f"{model} did not start: {sw} (a higher tier's model) took the card while this request waited"))
            _lock.wait(poll_s)
        if _switching_to == model:
            _switching_to = None
    out["waited_s"] = round(time.time() - t0, 2)
    if out["other_inflight_at_start"]:
        _write_state()
    on = [m for m in MODELS if loaded(m)]
    if FULL and model not in on:
        t1 = time.time()
        ok, how = _load(model)
        after = _fresh_running()
        left = sorted(m for m in MODELS if m != model and m in after)
        out["swap"] = {"from": sorted(on), "to": model, "load_s": round(time.time() - t1, 1), "ok": ok,
                       "how": how, "left_loaded": left}
        _last_swap = dict(out["swap"], at=time.time())
        print(f"  tier model swap: {sorted(on) or 'nothing'} -> {model} in {out['swap']['load_s']} s ({how})"
              + (f"; STILL LOADED: {left}" if left else ""), flush=True)
    return out


def snapshot() -> dict:
    with _lock:
        return {"enabled": ENABLED, "main": MAIN, "max": MAX, "models": list(MODELS),
                "tiers": {t: TABLE.model_for(t) for t in tier_models.ORDER},
                "inflight": dict(_inflight), "switching_to": _switching_to, "last_end": dict(_last_end),
                "last_max_end": _last_end.get(MAX, 0.0) if MAX else 0.0, "idle_s": IDLE_S,
                "last_swap": dict(_last_swap), "table": TABLE.source}


def _reset_for_tests() -> None:
    global _switching_to, _running_cache, _last_swap
    with _lock:
        _inflight.clear()
        _switching_to = None
        _last_end.clear()
        _last_swap = {}
        _running_cache = (0.0, set())
