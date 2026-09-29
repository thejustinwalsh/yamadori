#!/usr/bin/env python
"""Max mode (mcp/max_mode.py; docs/FLASH-NEXT.md sections 4-5), offline.

WHAT IS GATED

  1. OFF BY DEFAULT: with YAMADORI_MAX_MODEL unset every request goes to the main model, nothing is refused or
     blocked, the one door and the budget behave as before.
  2. THE ROUTING: tier max -> the max model; every other tier -> the main model; a side call while the max model is
     loaded -> the max model (never refused).
  3. THE CAPACITY REFUSAL: a non-max request while max holds the card is 503 `model_at_capacity`, OpenAI-shaped,
     with Retry-After -- through the server (blocking and streamed) and through api_errors for the one door's
     backstop.
  4. THE SWAP: a max request that arrives while main work is in flight marks the switch (new main work refused) and
     WAITS for that work (never cancelling it); the return to the main model happens on the next non-max request once
     no max request is in flight (IDLE_S unset), or after the operator's IDLE_S.
  5. INHERITANCE: the model bound to a request's cancel token is what model.shape, shomen, the decider and vision
     use, on job threads too; nothing reaches /upstream/<main> while the max model holds the card (model.post,
     model.props, proxy._upstream_json, tiers.accepted_efforts, idle.stack_idle, the worker).
  6. THE DECIDER PER MODEL: label spellings and label priors are kept per model.
  7. THINKING AT MAX follows the model's card: no Bonsai cap (step cap, helper cap) at max; unchanged elsewhere.

NO NETWORK: llama-swap's /running is stubbed (max_mode.running), urllib.request.urlopen fails the test if reached
where it must not be, and every store is a temp path set before import.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_max_mode_")
os.environ["YAMADORI_MAX_STATE"] = os.path.join(_TMP, "max_mode.json")
os.environ.setdefault("YAMADORI_ACCOUNTS_DIR", os.path.join(_TMP, "accounts"))
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
os.environ.pop("YAMADORI_MAX_MODEL", None)
os.environ.pop("YAMADORI_TIER_MODELS", None)
os.environ.pop("YAMADORI_MAX_IDLE_S", None)

import cancel  # noqa: E402
import max_mode  # noqa: E402
import api_errors  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, str(detail)))
    return bool(ok)


_LOADED: set[str] = set()


_LOADS: list[str] = []
_LEAK: set[str] = set()


def _fake_load(model, timeout=0):
    """llama-swap's group swap, simulated: the model starts and every other main model stops -- except the ones a
    test marks as leaking (a reload leak must be reported, never silent)."""
    _LOADS.append(model)
    for m in list(_LOADED):
        if m != model and m in max_mode.MODELS and m not in _LEAK:
            _LOADED.discard(m)
    _LOADED.add(model)
    return True, "HTTP 200 (fake)"


def reload(max_model: str | None, idle_s: str | None = None, table: str | None = None):
    """max_mode with this configuration; llama-swap's /running is _LOADED and a load is _fake_load (NEVER the live
    llama-swap: 2026-09-29 an unstubbed load from this suite started flash-next on the running stack)."""
    if max_model:
        os.environ["YAMADORI_MAX_MODEL"] = max_model
    else:
        os.environ.pop("YAMADORI_MAX_MODEL", None)
    if table:
        os.environ["YAMADORI_TIER_MODELS"] = table
    else:
        os.environ.pop("YAMADORI_TIER_MODELS", None)
    if idle_s is not None:
        os.environ["YAMADORI_MAX_IDLE_S"] = idle_s
    else:
        os.environ.pop("YAMADORI_MAX_IDLE_S", None)
    importlib.reload(max_mode)
    max_mode.running = lambda: set(_LOADED)
    max_mode._load = _fake_load
    max_mode._reset_for_tests()
    _LOADS.clear()
    _LEAK.clear()
    try:
        os.remove(os.environ["YAMADORI_MAX_STATE"])
    except OSError:
        pass
    return max_mode


class NoNetwork:
    """urlopen fails the test (records the URL) while active."""

    def __init__(self):
        self.urls: list[str] = []

    def __enter__(self):
        self.saved = urllib.request.urlopen

        def refuse(req, *a, **k):
            url = req if isinstance(req, str) else req.full_url
            self.urls.append(url)
            raise AssertionError(f"network reached: {url}")
        urllib.request.urlopen = refuse
        return self

    def __exit__(self, *a):
        urllib.request.urlopen = self.saved
        return False


def bound(model: str | None):
    tok = cancel.Token()
    ctx = cancel.bound(tok)
    ctx.__enter__()
    if model:
        max_mode.set_current(model)
    return tok, ctx


# ================================================================= 1. off ====
def test_off_by_default():
    reload(None)
    _LOADED.clear()
    check(max_mode.ENABLED is False, "off: YAMADORI_MAX_MODEL unset -> max mode off")
    d = max_mode.decide("max", False)
    check(d.model == "bonsai" and not d.refuse, "off: tier max goes to the main model", d.record())
    _LOADED.add("flash-next")
    check(not max_mode.blocks("bonsai"), "off: nothing is blocked, even with another model loaded")
    with NoNetwork() as nn:
        max_mode.guard("bonsai")
        check(not nn.urls, "off: the guard touches nothing")
    import tiers
    import budget
    budget._POOL = 262144
    b = tiers.budget(1000, True, role="main", step_cap=6144)
    check(b["reasoning_budget_tokens"] == 6144, "off: the step cap binds as before", b["reasoning_budget_tokens"])
    _LOADED.clear()


# ============================================================= 2. routing ====
def test_routing():
    reload("flash-next")
    _LOADED.clear()
    check(max_mode.ENABLED and max_mode.MAX == "flash-next", "on: YAMADORI_MAX_MODEL=flash-next")
    d = max_mode.decide("max", False)
    check(d.model == "flash-next" and not d.refuse, "tier max -> flash-next", d.record())
    d = max_mode.decide("medium", False)
    check(d.model == "bonsai" and not d.refuse, "tier medium with nothing of max active -> bonsai", d.record())
    d = max_mode.decide("medium", True)
    check(d.model == "bonsai", "a side call with the max model not loaded -> bonsai", d.record())
    _LOADED.add("flash-next")
    d = max_mode.decide("medium", True)
    check(d.model == "flash-next" and not d.refuse,
          "a side call while flash-next is loaded -> flash-next, never refused", d.record())
    d = max_mode.decide("xhigh", False)
    check(d.model == "bonsai" and not d.refuse,
          "max idle (no max in flight, IDLE_S unset): a non-max request swaps back to bonsai", d.record())
    _LOADED.clear()


# ============================================================ 3. capacity ====
def test_capacity_refusal():
    reload("flash-next")
    _LOADED.add("flash-next")
    lease = max_mode.Lease(max_mode.decide("max", False))
    d = max_mode.decide("medium", False)
    check(d.refuse and d.model == "bonsai", "a non-max request while a max request is in flight is refused",
          d.record())
    check(d.retry_after == max_mode.RETRY_AFTER_UNKNOWN,
          "Retry-After while max work is in flight: api_errors' existing 503 value (the end is unknown)",
          d.retry_after)
    err = api_errors.at_capacity(max_mode.MAX, d.retry_after, d.why)
    body = err.body()["error"]
    check(err.status == 503 and err.type == "service_unavailable_error" and body["code"] == "model_at_capacity"
          and all(k in body for k in ("message", "type", "param", "code")),
          "503 model_at_capacity in OpenAI's four fields", body)
    check(err.headers.get("Retry-After") == "30" and body.get("retryable") is True,
          "Retry-After header and retryable", err.headers)
    check("max mode" in err.message and "flash-next" in err.message and "retry in about 30 s" in err.message,
          "the message says max mode and when (without a table file: the 2026-09-28 text word for word)",
          err.message)
    lease.release()
    check(not max_mode.decide("medium", False).refuse, "released: the next non-max request is served")
    # the operator's idle period
    reload("flash-next", idle_s="600")
    _LOADED.add("flash-next")
    lease = max_mode.Lease(max_mode.decide("max", False))
    lease.release()
    d = max_mode.decide("medium", False)
    check(d.refuse and 590 <= d.retry_after <= 600,
          "with IDLE_S=600 just after max ended: refused, Retry-After = the rest of the idle period",
          d.retry_after)
    max_mode._last_end["flash-next"] = time.time() - 601
    check(not max_mode.decide("medium", False).refuse, "after IDLE_S: served (the swap back)")
    # the one door's backstop maps to the same error
    e = max_mode.ModelAtCapacity("bonsai")
    ae = api_errors.of_exception(e)
    check(ae.status == 503 and ae.code == "model_at_capacity", "ModelAtCapacity -> 503 model_at_capacity",
          (ae.status, ae.code))
    _LOADED.clear()


# ================================================================ 4. swap ====
def test_swap_waits_never_cancels():
    reload("flash-next")
    _LOADED.clear()
    main_lease = max_mode.Lease(max_mode.decide("medium", False))
    main_token = cancel.Token()
    max_lease = max_mode.Lease(max_mode.decide("max", False))
    check(max_mode.snapshot()["switching_to"] == "flash-next",
          "a max request while bonsai work is in flight marks the switch")
    check(max_mode.decide("low", False).refuse, "during the switch new bonsai work is refused")
    out: dict = {}

    def maxturn():
        tok = cancel.Token()
        with cancel.bound(tok):
            out.update(max_mode.wait_ready("flash-next"))
    th = threading.Thread(target=maxturn)
    th.start()
    time.sleep(0.8)
    check(th.is_alive(), "the max request waits while bonsai's work is in flight")
    check(not main_token.cancelled, "and the bonsai work is not cancelled")
    main_lease.release()
    th.join(5)
    check(not th.is_alive() and out.get("waited_s", 0) >= 0.5 and out.get("other_inflight_at_start") == 1,
          "released: the max request proceeds, having waited", out)
    check("swap" not in out and _LOADS == [],
          "without a table file its first upstream call loads flash-next (no explicit swap; the table's is gated in "
          "test_table_capacity_and_preemption)", out)
    check(max_mode.snapshot()["switching_to"] is None, "the switch is done once the max request runs")
    max_lease.release()
    # a cancelled waiter stops waiting
    main_lease = max_mode.Lease(max_mode.decide("medium", False))
    max_lease = max_mode.Lease(max_mode.decide("max", False))
    err: dict = {}

    def cancelled_turn():
        tok = cancel.Token()
        tok.cancel("client went away")
        with cancel.bound(tok):
            try:
                max_mode.wait_ready("flash-next")
            except cancel.Cancelled as e:
                err["e"] = e
    th = threading.Thread(target=cancelled_turn)
    th.start()
    th.join(5)
    check("e" in err, "a waiting max request whose client hangs up stops waiting (Cancelled)")
    max_lease.release()
    main_lease.release()
    st = json.load(open(os.environ["YAMADORI_MAX_STATE"], encoding="utf-8"))
    check(st.get("inflight") == {"bonsai": 0, "flash-next": 0} and st.get("switching_to") is None,
          "the state file other processes read is written and ends clean", st)


# ======================================================== 5. inheritance ====
def test_inheritance_and_no_reload():
    reload("flash-next")
    _LOADED.clear()
    _LOADED.add("flash-next")
    import model
    import tiers
    import budget
    budget._POOL = 262144
    importlib.reload(model)
    # flash-next's served template set, pinned (a request bound to it reads its /props once: the live stack)
    tiers._accepted_of["flash-next"] = tiers.FALLBACK_EFFORTS
    tok, ctx = bound("flash-next")
    try:
        check(max_mode.current() == "flash-next", "the request's model is bound to its cancel token")
        seen: dict = {}

        def job():
            with cancel.bound(tok):          # how job threads re-bind (proxy._in_thread, shomen)
                seen["m"] = max_mode.current()
                seen["shape"] = model.shape({"messages": [{"role": "user", "content": "x"}]}, "medium")["model"]
                import shomen
                seen["shomen"] = shomen._model()
                import vision
                seen["vision"] = vision._vision_model()
        th = threading.Thread(target=job)
        th.start()
        th.join(5)
        check(seen.get("m") == "flash-next" and seen.get("shape") == "flash-next",
              "a job thread re-binding the token inherits flash-next; the one door shapes for it", seen)
        check(seen.get("shomen") == "flash-next" and seen.get("vision") == "flash-next",
              "the second brain and yama_describe_image follow the conversation's model", seen)
    finally:
        ctx.__exit__(None, None, None)
    # nothing touches bonsai while flash-next is loaded
    with NoNetwork() as nn:
        try:
            model.post({"model": "bonsai", "messages": []})
            check(False, "model.post refuses bonsai while flash-next holds the card")
        except max_mode.ModelAtCapacity:
            check(True, "model.post refuses bonsai while flash-next holds the card (before any request)")
        try:
            model.props("bonsai")
            check(False, "model.props refuses bonsai")
        except max_mode.ModelAtCapacity:
            check(True, "model.props refuses bonsai (a /props read would load it)")
        r = model.release_slot(1, model="bonsai")
        check(r.get("skipped") and not r.get("ok"), "a bonsai slot release is skipped", r)
        tiers._accepted_of.clear()
        check(tiers.accepted_efforts() == tiers.FALLBACK_EFFORTS,
              "accepted_efforts outside a request does not read bonsai's /props", tiers.accepted_efforts())
        import proxy
        try:
            proxy._upstream_json("/upstream/bonsai/tokenize", {"content": "x"})
            check(False, "proxy._upstream_json refuses bonsai")
        except max_mode.ModelAtCapacity:
            check(True, "proxy._upstream_json refuses /upstream/bonsai/... (warms, tokenize, apply-template)")
        rec: dict = {}
        proxy._warm_now("k", "bonsai", [], {}, rec)
        check(rec.get("state") == "skipped" and "max mode" in (rec.get("why") or ""),
              "a bonsai conversation's warm is skipped", rec)
        import vision
        check(vision.main_sees(refresh=True) is False, "vision.main_sees does not read bonsai's /props")
        check(not nn.urls, "no URL was reached", nn.urls)
    # ... unless this thread works for a request admitted to bonsai (the swap back)
    tok, ctx = bound("bonsai")
    try:
        max_mode.guard("bonsai")
        check(True, "a request admitted to bonsai may load it (the swap back)")
    except max_mode.ModelAtCapacity:
        check(False, "a request admitted to bonsai may load it (the swap back)")
    finally:
        ctx.__exit__(None, None, None)
    # background callers
    import idle
    st = idle.stack_idle()
    check(st["idle"] is False and "max mode" in st["why"], "idle.stack_idle: not idle while max holds the card",
          st["why"])
    _LOADED.clear()


def test_worker_defers():
    reload("flash-next")
    _LOADED.clear()
    _LOADED.add("flash-next")
    import worker
    import jobs
    calls: dict = {}

    def handler(job, ctx):
        import model
        model.post({"model": "bonsai", "messages": []})
    saved = dict(worker.HANDLERS)
    saved_defer = jobs.defer
    jobs.defer = lambda jid, until, why: calls.update(jid=jid, until=until, why=why) or "queued"
    worker.HANDLERS["maxtest"] = handler
    try:
        with NoNetwork() as nn:
            st = worker.run_one({"id": "j1", "queue": "maxtest", "payload": {}})
        check(st == "queued" and "max mode" in calls.get("why", "") and calls.get("until", 0) > time.time(),
              "a worker job that asks bonsai under max mode is deferred without an attempt", calls)
        check(not nn.urls, "and reached nothing", nn.urls)
    finally:
        worker.HANDLERS.clear()
        worker.HANDLERS.update(saved)
        jobs.defer = saved_defer
    _LOADED.clear()


# ============================================================ 6. decider ====
def test_decider_per_model():
    reload("flash-next")
    import decider_bonsai as D
    import decide_turn
    D._SPELL_IDS.clear()
    asked: list[str] = []

    def up(path, payload=None, timeout=30):
        asked.append(max_mode.current())
        return {"tokens": [{"id": 65}]}
    tok, ctx = bound("bonsai")
    try:
        D.spelling_ids("A", upstream=up)
        k1 = decide_turn._cf_key("x")
    finally:
        ctx.__exit__(None, None, None)
    tok, ctx = bound("flash-next")
    try:
        D.spelling_ids("A", upstream=up)
        k2 = decide_turn._cf_key("x")
    finally:
        ctx.__exit__(None, None, None)
    check(set(D._SPELL_IDS) >= {"bonsai:A", "flash-next:A"} and asked.count("flash-next") >= 1,
          "the label check runs per model (each model's own /tokenize, kept per model)", sorted(D._SPELL_IDS))
    check(k1 != k2 and k2.startswith("flash-next:"), "the label priors are keyed per model", (k1, k2))
    reload(None)
    check(decide_turn._cf_key("x") == "x", "off: the prior keys are unchanged")
    D._SPELL_IDS.clear()


# ============================================================ 7. thinking ====
def test_thinking_at_max_follows_the_card():
    reload("flash-next")
    import tiers
    import budget
    budget._POOL = 262144
    tok, ctx = bound("flash-next")
    try:
        b = tiers.budget(1000, True, role="main", step_cap=6144)
        h = tiers.budget(1000, True, role="helper", step_cap=3072)
        c = tiers.budget(1000, True, role="main", cap=1200)
    finally:
        ctx.__exit__(None, None, None)
    check(b["reasoning_budget_tokens"] > 6144, "at max: no agent-step/user-turn cap (the whole room)",
          b["reasoning_budget_tokens"])
    check(h["reasoning_budget_tokens"] > 3072, "at max: no second-brain job cap", h["reasoning_budget_tokens"])
    check(c["reasoning_budget_tokens"] == 1200, "at max: a benchmark's reasoning_cap still wins",
          c["reasoning_budget_tokens"])
    tok, ctx = bound("bonsai")
    try:
        b2 = tiers.budget(1000, True, role="main", step_cap=6144)
    finally:
        ctx.__exit__(None, None, None)
    check(b2["reasoning_budget_tokens"] == 6144, "on bonsai the caps bind as before", b2["reasoning_budget_tokens"])


# ============================================================= 8. server ====
def test_server_admission():
    reload("flash-next")
    _LOADED.clear()
    import accounts
    import proxy
    import server
    importlib.reload(server)           # bind the reloaded max_mode
    from starlette.testclient import TestClient
    key = accounts.create("max-mode-tests")
    client = TestClient(server.app)
    h = {"Authorization": f"Bearer {key}"}
    seen: list[dict] = []

    def fake_complete(body):
        seen.append({"model": body.get("_upstream_model"), "cap": body.get("_capacity"),
                     "inflight": dict(max_mode.snapshot()["inflight"])})
        return {"id": "x", "object": "chat.completion", "model": "yamadori",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}]}

    def fake_stream(body, public_name="yamadori", token=None):
        seen.append({"model": body.get("_upstream_model"), "stream": True,
                     "inflight": dict(max_mode.snapshot()["inflight"])})
        yield b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
        yield b"data: [DONE]\n\n"
    saved = (proxy.complete, proxy.stream_body)
    proxy.complete, proxy.stream_body = fake_complete, fake_stream
    try:
        msg = [{"role": "user", "content": "write a function that adds two numbers"}]
        r = client.post("/v1/chat/completions", json={"model": "yamadori", "messages": msg,
                                                      "reasoning_effort": "max"}, headers=h)
        check(r.status_code == 200 and seen[-1]["model"] == "flash-next"
              and seen[-1]["inflight"].get("flash-next") == 1,
              "server: a max request is admitted to flash-next with a lease", (r.status_code, seen[-1:]))
        check(max_mode.snapshot()["inflight"].get("flash-next") == 0, "server: the lease ends with the request")
        # hold a max lease: a medium request is refused, blocking and streamed
        lease = max_mode.Lease(max_mode.decide("max", False))
        r = client.post("/v1/chat/completions", json={"model": "yamadori", "messages": msg,
                                                      "reasoning_effort": "medium"}, headers=h)
        body = r.json()
        check(r.status_code == 503 and body["error"]["code"] == "model_at_capacity"
              and r.headers.get("retry-after") == "30",
              "server: medium while max is in flight -> 503 model_at_capacity + Retry-After", (r.status_code, body))
        with client.stream("POST", "/v1/chat/completions", json={
                "model": "yamadori", "messages": msg, "reasoning_effort": "low", "stream": True}, headers=h) as rs:
            text = "".join(rs.iter_text())
            check(rs.status_code == 503 and "model_at_capacity" in text,
                  "server: a streamed low request is refused before any byte", (rs.status_code, text[:120]))
        lease.release()
        with client.stream("POST", "/v1/chat/completions", json={
                "model": "yamadori", "messages": msg, "reasoning_effort": "max", "stream": True}, headers=h) as rs:
            "".join(rs.iter_text())
        check(seen[-1].get("stream") and seen[-1]["model"] == "flash-next"
              and max_mode.snapshot()["inflight"].get("flash-next") == 0,
              "server: a streamed max request holds its lease until the stream ends", seen[-1])
        r = client.post("/v1/chat/completions", json={"model": "yamadori", "messages": msg,
                                                      "reasoning_effort": "medium"}, headers=h)
        check(r.status_code == 200 and seen[-1]["model"] == "bonsai",
              "server: with no max in flight a medium request goes to bonsai (the swap back)", seen[-1])
        _LOADED.add("flash-next")
        title = [{"role": "system", "content": "Generate a short title for this conversation. Reply with the "
                                              "title only."},
                 {"role": "user", "content": "how do I sort a list in python"}]
        r = client.post("/v1/chat/completions", json={"model": "yamadori", "messages": title}, headers=h)
        util = max_mode.is_utility({"messages": title})
        check(util and r.status_code == 200 and seen[-1]["model"] == "flash-next",
              "server: a side call while flash-next is loaded is served there, not refused",
              (util, seen[-1]))
    finally:
        proxy.complete, proxy.stream_body = saved
        _LOADED.clear()


# ====================================================== 9. the tier table ====
# ONE MODEL PER EFFORT TIER (operator, 2026-09-29): minimal..high -> bonsai, xhigh -> mirai-s, max -> flash-next,
# read from the table (mcp/tier_models.yaml, here a temp copy with mirai-s's window filled as the deploy writes it).
def _table_file(mirai_vision=False) -> str:
    import yaml
    src = os.path.join(os.path.dirname(HERE), "mcp", "tier_models.yaml")
    spec = yaml.safe_load(open(src, encoding="utf-8"))
    spec["models"]["mirai-s"]["window"] = {"ctx": 98304, "slots": 3, "main_cap": 95232, "source": "test"}
    spec["models"]["mirai-s"]["vision"] = mirai_vision
    p = os.path.join(_TMP, "tier_models.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f)
    return p


def test_table_routing_and_ranks():
    reload(None, table=_table_file())
    _LOADED.clear()
    check(max_mode.ENABLED and max_mode.MODELS == ["bonsai", "mirai-s", "flash-next"] and max_mode.MAX == "flash-next",
          "table: three main models in rank order; max mode's name is the max row", max_mode.MODELS)
    got = {t: max_mode.decide(t, False).model for t in ("minimal", "low", "medium", "high", "xhigh", "max")}
    check(got == {"minimal": "bonsai", "low": "bonsai", "medium": "bonsai", "high": "bonsai", "xhigh": "mirai-s",
                  "max": "flash-next"}, "table: each tier goes to its model, nothing refused on an idle card", got)
    check([max_mode.rank(m) for m in max_mode.MODELS] == [3, 4, 5], "table: ranks are the highest tier each serves")
    reload(None, table="xhigh=mirai-s:medium,max=flash-next")
    import tier_models
    check(max_mode.decide("xhigh", False).model == "mirai-s" and tier_models.value("mirai-s", "effort") == "medium"
          and tier_models.table().source.startswith("YAMADORI_TIER_MODELS"),
          "table: the inline form (tier=model[:effort]) for experiments")
    os.environ["YAMADORI_MAX_MODEL"] = "flash-next"
    reload("flash-next")
    check(max_mode.MODELS == ["bonsai", "flash-next"] and max_mode.decide("xhigh", False).model == "bonsai",
          "legacy: YAMADORI_MAX_MODEL alone is the one-row table {max: flash-next}")


def test_table_capacity_and_preemption():
    reload(None, table=_table_file())
    _LOADED.clear()
    _LOADED.add("mirai-s")
    lease = max_mode.Lease(max_mode.decide("xhigh", False))
    d = max_mode.decide("medium", False)
    check(d.refuse and d.holder == "mirai-s", "a medium request while mirai-s (xhigh) works is refused 503",
          d.record())
    err = api_errors.at_capacity(d.holder, d.retry_after, d.why, model=d.model, tier="medium")
    check(err.status == 503 and 'reasoning_effort "medium"' in err.message and "keeps its model" in err.message,
          "the client is told plainly: its tier, that another tier's model holds the card, retry, it keeps its model",
          err.message)
    d = max_mode.decide("max", False)
    check(not d.refuse and d.model == "flash-next", "a max request while mirai-s works is admitted (it outranks it)",
          d.record())
    lease_max = max_mode.Lease(d)
    check(max_mode.snapshot()["switching_to"] == "flash-next", "... and marks the switch")
    check(max_mode.decide("xhigh", False).refuse, "during the switch a new xhigh request is refused")
    u = max_mode.decide("medium", True)
    check(not u.refuse and u.model == "flash-next", "a side call is served by the model taking the card, not refused",
          u.record())
    lease.release()
    out = max_mode.wait_ready("flash-next")
    check(out["swap"]["from"] == ["mirai-s"] and out["swap"]["to"] == "flash-next"
          and out["swap"]["left_loaded"] == [], "the swap mirai-s -> flash-next, nothing left loaded", out)
    lease_max.release()
    d = max_mode.decide("medium", False)
    check(not d.refuse and d.model == "bonsai" and "swap from flash-next" in d.why,
          "nothing in flight: a medium request swaps bonsai back", d.why)
    lease_b = max_mode.Lease(d)
    out = max_mode.wait_ready("bonsai")
    check(out["swap"]["to"] == "bonsai" and _LOADED & set(max_mode.MODELS) == {"bonsai"},
          "and loads it (both ways clean)", out)
    # PREEMPTION: bonsai works; an xhigh request waits; a max request overtakes it before it starts
    got: dict = {}

    def xhigh_turn():
        tok = cancel.Token()
        with cancel.bound(tok):
            try:
                got["out"] = max_mode.wait_ready("mirai-s")
            except max_mode.ModelAtCapacity as e:
                got["refused"] = str(e)
    lease_x = max_mode.Lease(max_mode.decide("xhigh", False))
    th = threading.Thread(target=xhigh_turn)
    th.start()
    time.sleep(0.6)
    lease_f = max_mode.Lease(max_mode.decide("max", False))
    th.join(5)
    check("refused" in got and "flash-next" in got["refused"],
          "a waiting xhigh request overtaken by a max request is refused (it never ran); no deadlock", got)
    lease_x.release()
    got2: dict = {}

    def max_turn():
        tok = cancel.Token()
        with cancel.bound(tok):
            got2["out"] = max_mode.wait_ready("flash-next")
    th = threading.Thread(target=max_turn)
    th.start()
    time.sleep(0.6)
    check(th.is_alive(), "the max request waits for bonsai's work (never cancels it)")
    lease_b.release()
    th.join(5)
    check((got2.get("out") or {}).get("swap", {}).get("to") == "flash-next", "then swaps", got2)
    lease_f.release()
    # A RELOAD LEAK is reported, not silent
    _LEAK.add("flash-next")
    lease_b = max_mode.Lease(max_mode.decide("medium", False))
    out = max_mode.wait_ready("bonsai")
    check(out["swap"]["left_loaded"] == ["flash-next"] and max_mode.snapshot()["last_swap"]["left_loaded"],
          "a model the group swap did not stop is recorded (left_loaded), in x_yamadori.capacity and the snapshot",
          out)
    lease_b.release()
    _LOADED.clear()


def test_table_profile_applied():
    reload(None, table=_table_file())
    import tiers
    import budget
    budget._POOL = 262144
    budget._LINE = None
    tiers._accepted_of.update({m: tiers.FALLBACK_EFFORTS for m in max_mode.MODELS})
    tier_x = tiers.resolve({"reasoning_effort": "xhigh"})
    tier_m = tiers.resolve({"reasoning_effort": "medium"})
    body = {"model": "yamadori", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64,
            "temperature": 0.2, "_client": True}
    tok, ctx = bound("bonsai")
    try:
        u = tiers.apply(body, tier_m, step_cap=tiers.USER_TURN_THINKING)
        a = tiers.apply(body, tier_m, step_cap=tiers.AGENT_STEP_THINKING, nudge=tiers.AGENT_STEP_NUDGE_MESSAGE)
        off = tiers.apply(body, tiers.resolve({"reasoning_effort": "minimal"}))
    finally:
        ctx.__exit__(None, None, None)
    check(u["reasoning_budget_tokens"] == 20480 and u["reasoning_effort"] == "medium" and u["max_tokens"] >= 24576
          and u["_answer"] == 2048 and u["temperature"] == 1.0,
          "bonsai, a user turn: medium, 20,480 thinking, turn >= 24,576, answer 2,048 whatever the client's 64, "
          "vendor sampling over the client's 0.2", {k: u.get(k) for k in ("reasoning_budget_tokens", "max_tokens",
                                                                          "_answer", "reasoning_effort")})
    check(a["reasoning_budget_tokens"] == 6144 and a["reasoning_budget_nudge"] == tiers.AGENT_STEP_NUDGE_MESSAGE,
          "bonsai, an agent step: 6,144 and the agent-step nudge", a["reasoning_budget_tokens"])
    check(off["max_tokens"] >= 24576 and off["enable_thinking"] is False and off["temperature"] == 0.7,
          "bonsai, thinking off (minimal): the turn is the room, never below the floor; instruct sampling",
          off["max_tokens"])
    rec = u["_sampling"]["profile"]
    check(rec["model"] == "bonsai" and rec["route"] == "user_turn" and rec["client"]["max_tokens"] == 64
          and rec["effort"]["class"] == "measurement" and "ada-surgery" in rec["effort"]["source"]
          and rec["turn_min_tokens"]["value"] == 24576,
          "x_yamadori.sampling.profile: the model, the route, each value's class and source, what the client asked",
          json.dumps(rec)[:400])
    tok, ctx = bound("mirai-s")
    try:
        mu = tiers.apply(body, tier_x, step_cap=tiers.USER_TURN_THINKING)
        forced = tiers.apply(body, tiers.resolve({"reasoning_effort": "xhigh"}, {"effort": "low"}),
                             step_cap=tiers.USER_TURN_THINKING)
        w = budget.budgets()
    finally:
        ctx.__exit__(None, None, None)
    check(mu["reasoning_effort"] == "xhigh" and mu["reasoning_budget_tokens"] > 20480
          and mu["_sampling"]["profile"]["effort"]["class"] == "vendor",
          "mirai-s: the card's effort (xhigh, labelled vendor) and no Bonsai cap (the room)",
          (mu["reasoning_effort"], mu["reasoning_budget_tokens"]))
    est = tiers.estimate_prompt_tokens(body)
    check(mu["reasoning_budget_tokens"] == 95232 - est - 2048 and mu["max_tokens"] == 95232 - est
          and mu["_sampling"]["profile"]["thinking_cap"]["class"] == "derived",
          "mirai-s: NO CHOSEN CAP (operator 2026-09-29) -- thinking = its window - prompt - answer, the turn = the "
          "window - prompt; the record labels it derived", (mu["reasoning_budget_tokens"], mu["max_tokens"], est))
    check(forced["reasoning_effort"] == "low" and forced["_sampling"]["profile"]["effort"]["from"] == "header",
          "a benchmark's header effort still wins (how the paired effort set varies one thing)")
    check(w["main"] == 95232 and w["pool"] == 98304 and w.get("model_window"),
          "mirai-s's own window (the table's, as the deploy writes it), not bonsai's", {k: w[k] for k in ("pool", "main")})
    body_nc = dict(body)
    body_nc.pop("_client")
    tok, ctx = bound("bonsai")
    try:
        internal = tiers.apply(dict(body_nc, max_tokens=5000), tier_m)
    finally:
        ctx.__exit__(None, None, None)
    check(internal["_answer"] == 5000, "an internal caller's answer allowance is still its own (only a client's is "
          "not trusted)", internal["_answer"])
    # rebudget keeps the profile
    tok, ctx = bound("bonsai")
    try:
        rb = tiers.rebudget(u)
    finally:
        ctx.__exit__(None, None, None)
    check(rb["reasoning_budget_tokens"] == 20480 and rb["max_tokens"] >= 24576
          and rb["reasoning_budget_message"] == u["reasoning_budget_message"],
          "a later hop (rebudget) keeps the profile's cap, floor and force-close")


def test_table_windows_and_vision():
    reload(None, table=_table_file(mirai_vision=False))
    import budget
    import catalog
    import vision
    budget._POOL = 262144
    budget._LINE = None
    wins = {t: catalog.tier_window(t) for t in ("medium", "xhigh", "max")}
    check(wins["xhigh"] == 95232 and wins["max"] == 259072 and wins["medium"] == budget.budgets(model="bonsai")["main"],
          "the advertised window is per model: each tier's own", wins)
    card = catalog._chat_card(wins["medium"])
    tb = card["x_yamadori"]["reasoning_effort"].get("tokens_by_value") or {}
    check(tb.get("xhigh") == 95232 and tb.get("max") == 259072 and "mirai" not in json.dumps(card),
          "the card lists each effort's window, model names kept internal", tb)
    tok, ctx = bound("mirai-s")
    try:
        vm = vision._vision_model()
    finally:
        ctx.__exit__(None, None, None)
    check(vm == "bonsai-vision", "mirai-s without its own projector: images go to the A4000 copy", vm)
    tok, ctx = bound("flash-next")
    try:
        vf = vision._vision_model()
    finally:
        ctx.__exit__(None, None, None)
    check(vf == "flash-next", "flash-next looks itself (its entry carries --mmproj)", vf)


def test_table_server_records():
    reload(None, table=_table_file())
    _LOADED.clear()
    import accounts
    import proxy
    import server
    importlib.reload(server)
    from starlette.testclient import TestClient
    key = accounts.create("tier-table-tests")
    client = TestClient(server.app)
    h = {"Authorization": f"Bearer {key}"}
    seen: list[dict] = []

    def fake_complete(body):
        seen.append({"model": body.get("_upstream_model"), "cap": body.get("_capacity"),
                     "client": body.get("_client")})
        return {"id": "x", "object": "chat.completion", "model": "yamadori",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}]}
    saved = proxy.complete
    proxy.complete = fake_complete
    try:
        msg = [{"role": "user", "content": "write a function that adds two numbers"}]
        r = client.post("/v1/chat/completions", json={"model": "yamadori", "messages": msg,
                                                      "reasoning_effort": "xhigh"}, headers=h)
        cap = seen[-1]["cap"] or {}
        check(r.status_code == 200 and seen[-1]["model"] == "mirai-s" and cap.get("model") == "mirai-s"
              and "tier xhigh is served by mirai-s" in (cap.get("why") or "") and cap.get("effort") == "xhigh"
              and seen[-1]["client"] is True,
              "server: xhigh -> mirai-s; x_yamadori.capacity says which model and why, and the profile's effort", cap)
        lease = max_mode.Lease(max_mode.decide("xhigh", False))
        r = client.post("/v1/chat/completions", json={"model": "yamadori", "messages": msg,
                                                      "reasoning_effort": "high"}, headers=h)
        body = r.json()
        check(r.status_code == 503 and body["error"]["code"] == "model_at_capacity"
              and body["error"].get("holder") == "mirai-s" and body["error"].get("tier") == "high",
              "server: high while xhigh holds the card -> 503, the holder and the asked tier in the error", body)
        lease.release()
    finally:
        proxy.complete = saved
        _LOADED.clear()


def test_no_offline_swap():
    """2026-09-29: an unstubbed test of this module started flash-next on the running stack. The real loader sends
    nothing unless the process is the proxy (enable_swaps) or carries YAMADORI_ALLOW_SWAP=1, and never under the
    offline guard."""
    reload(None, table=_table_file())
    importlib.reload(max_mode)                  # the REAL _load, not the fake
    max_mode.running = lambda: set(_LOADED)
    with NoNetwork() as nn:
        ok, how = max_mode._load("flash-next")
        check(not ok and "not sent" in how and not nn.urls,
              "the real loader in a process that is not the proxy sends nothing", (ok, how, nn.urls))
        max_mode.enable_swaps()
        saved = os.environ.get("YAMADORI_OFFLINE_GUARD")
        os.environ["YAMADORI_OFFLINE_GUARD"] = "1"
        try:
            ok, how = max_mode._load("flash-next")
        finally:
            if saved is None:
                os.environ.pop("YAMADORI_OFFLINE_GUARD", None)
            else:
                os.environ["YAMADORI_OFFLINE_GUARD"] = saved
        check(not ok and "offline" in how and not nn.urls,
              "even with swaps enabled, an offline suite never swaps the card", (ok, how, nn.urls))
        max_mode.enable_swaps(False)
    reload(None)


def test_legacy_is_inert():
    """THE RUNNING STACK TODAY (YAMADORI_MAX_MODEL=flash-next, no YAMADORI_TIER_MODELS): a proxy restarted with this
    working tree must behave exactly as the 2026-09-28 max mode -- no token profile (the client's max_tokens is its
    answer allowance, flash-next thinks with no Bonsai cap, bonsai keeps its caps), no explicit swap, only bonsai is
    ever blocked, the same budgets and /v1/models card, the same 503 text."""
    reload("flash-next")
    _LOADED.clear()
    _LOADED.add("bonsai")
    import tiers
    import budget
    import catalog
    import tier_models
    budget._POOL = 262144
    budget._LINE = None
    tiers._accepted_of.update({m: tiers.FALLBACK_EFFORTS for m in ("bonsai", "flash-next")})
    check(max_mode.ENABLED and not max_mode.FULL and not tier_models.profiles_on(),
          "legacy: routing on, the table's profiles/swap/windows off")
    body = {"model": "yamadori", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5000,
            "temperature": 0.2, "_client": True}
    tm = tiers.resolve({"reasoning_effort": "medium"})
    tx = tiers.resolve({"reasoning_effort": "max"})
    tok, ctx = bound("bonsai")
    try:
        u = tiers.apply(body, tm, step_cap=tiers.USER_TURN_THINKING)
        a = tiers.apply(body, tm, step_cap=tiers.AGENT_STEP_THINKING, nudge=tiers.AGENT_STEP_NUDGE_MESSAGE)
        wb = budget.budgets()
    finally:
        ctx.__exit__(None, None, None)
    s_main = tiers._shares()["main"]
    check(u["_answer"] == 5000 and u["reasoning_budget_tokens"] == tiers.USER_TURN_THINKING
          and u["max_tokens"] == s_main - tiers.estimate_prompt_tokens(body) and u["reasoning_effort"] == "medium"
          and "profile" not in u["_sampling"] and "_profile" not in u,
          "legacy, bonsai user turn: the client's max_tokens is the answer, 20,480 thinking, no profile record",
          {k: u.get(k) for k in ("_answer", "reasoning_budget_tokens", "max_tokens")})
    check(a["reasoning_budget_tokens"] == tiers.AGENT_STEP_THINKING
          and a["reasoning_budget_nudge"] == tiers.AGENT_STEP_NUDGE_MESSAGE, "legacy, bonsai agent step: as before")
    tok, ctx = bound("flash-next")
    try:
        f = tiers.apply(body, tx, step_cap=tiers.USER_TURN_THINKING)
        wf = budget.budgets()
    finally:
        ctx.__exit__(None, None, None)
    check(f["reasoning_budget_tokens"] > tiers.USER_TURN_THINKING and f["reasoning_effort"] == "xhigh"
          and f["_answer"] == 5000, "legacy, flash-next at max: the card's room (no Bonsai cap), xhigh, as before",
          f["reasoning_budget_tokens"])
    check(wb == wf and "model_window" not in wf, "legacy: one pool for both models, as before", wf.get("main"))
    card = catalog._chat_card(catalog.context_window())
    check("tokens_by_value" not in card["x_yamadori"]["reasoning_effort"], "legacy: the /v1/models card unchanged")
    check(not max_mode.blocks("flash-next") and max_mode.blocks("bonsai") is False,
          "legacy: flash-next is never blocked; bonsai (loaded) is not")
    _LOADED.clear()
    _LOADED.add("flash-next")
    check(max_mode.blocks("bonsai"), "legacy: bonsai is blocked while flash-next is loaded, as before")
    _LOADED.clear()
    _LOADED.add("bonsai")
    lease = max_mode.Lease(max_mode.decide("max", False))
    out = max_mode.wait_ready("flash-next")
    check("swap" not in out and _LOADS == [], "legacy: no explicit swap (the request's first upstream call loads it)",
          out)
    lease.release()
    _LOADED.clear()


def main() -> int:
    for fn in (test_off_by_default, test_routing, test_capacity_refusal, test_swap_waits_never_cancels,
               test_inheritance_and_no_reload, test_worker_defers, test_decider_per_model,
               test_thinking_at_max_follows_the_card, test_server_admission,
               test_table_routing_and_ranks, test_table_capacity_and_preemption, test_table_profile_applied,
               test_table_windows_and_vision, test_table_server_records, test_no_offline_swap,
               test_legacy_is_inert):
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
            import traceback
            check(False, f"{fn.__name__} raised", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    reload(None)
    bad = [r for r in _results if not r[0]]
    for ok, name, detail in _results:
        print(("ok    " if ok else "FAIL  ") + name + (f"  ({detail[:300]})" if not ok and detail else ""))
    print(f"\n{'=' * 70}\n  {len(_results) - len(bad)}/{len(_results)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
