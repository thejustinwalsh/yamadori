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
  5. INHERITANCE: the model bound to a request's cancel token is what model.shape, the decider and vision
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
import re
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
            with cancel.bound(tok):          # how job threads re-bind (proxy._in_thread)
                seen["m"] = max_mode.current()
                seen["shape"] = model.shape({"messages": [{"role": "user", "content": "x"}]}, "medium")["model"]
                import vision
                seen["vision"] = vision._vision_model()
        th = threading.Thread(target=job)
        th.start()
        th.join(5)
        check(seen.get("m") == "flash-next" and seen.get("shape") == "flash-next",
              "a job thread re-binding the token inherits flash-next; the one door shapes for it", seen)
        check(seen.get("vision") == "flash-next",
              "yama_describe_image follows the conversation's model (the second "
              "brain that did too was removed 2026-09-29)", seen)
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
    # the UNLOCKED path, kept under test: mirai-s as it stood before its lane step (the live table locks it since
    # the 2026-10-01 gate: test_live_table_locks_every_main_model)
    spec["models"]["mirai-s"]["locked"] = False
    spec["models"]["mirai-s"]["helpers"] = {"decider": "self", "side_calls": "self"}
    p = os.path.join(_TMP, "tier_models.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f)
    return p


def test_live_table_locks_every_main_model():
    """The deployed table (bench/deploy_tier_models.py, 2026-10-01): mirai-s's gate measured its lane material
    (bench/results/mirai_s/gate-20261001-merged: ACTIVE below CLEARED at 8K/32K/64K, ~14-16%), so all three main
    models are locked at -np 1 and their jjava and side calls run on bonsai-a4000."""
    import yaml
    spec = yaml.safe_load(open(os.path.join(HERE, "tier_models.yaml"), encoding="utf-8"))["models"]
    got = {m: (spec[m].get("locked"), (spec[m].get("helpers") or {}).get("decider"),
               (spec[m].get("window") or {}).get("slots")) for m in ("bonsai", "mirai-s", "flash-next")}
    check(all(v[0] is True and v[1] == "bonsai-a4000" for v in got.values()) and got["mirai-s"][2] == 1,
          "the live table: bonsai, mirai-s and flash-next locked, jjava on bonsai-a4000; mirai-s one slot", got)


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
    check(not d.refuse and d.model == "bonsai-a4000" and d.holder == "mirai-s" and "other card" in d.why,
          "a medium request while mirai-s (xhigh) works is served on bonsai's other card, bonsai-a4000 "
          "(operator 2026-10-06: \"Serve it on the A4000\")", d.record())
    # a lower tier whose model has NO other card is still refused: xhigh (mirai-s) while flash-next works
    lease_fn = max_mode.Lease(max_mode.Decision("flash-next", "max", False))
    d = max_mode.decide("xhigh", False)
    lease_fn.release()
    check(d.refuse and d.holder in ("flash-next", "mirai-s"),
          "a lower tier with no other card (mirai-s) is still refused 503", d.record())
    err = api_errors.at_capacity(d.holder, d.retry_after, d.why, model=d.model, tier="xhigh")
    check(err.status == 503 and 'reasoning_effort "xhigh"' in err.message and "keeps its model" in err.message,
          "the client is told plainly: its tier, that another tier's model holds the card, retry, it keeps its model",
          err.message)
    d = max_mode.decide("max", False)
    check(not d.refuse and d.model == "flash-next", "a max request while mirai-s works is admitted (it outranks it)",
          d.record())
    lease_max = max_mode.Lease(d)
    check(max_mode.snapshot()["switching_to"] == "flash-next", "... and marks the switch")
    check(max_mode.decide("xhigh", False).refuse, "during the switch a new xhigh request is refused")
    u = max_mode.decide("medium", True)
    check(not u.refuse and u.model == "bonsai-a4000",
          "a side call while flash-next (LOCKED) takes the card goes to bonsai-a4000, not refused", u.record())
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
    check(mu["reasoning_effort"] == "xhigh" and mu["_sampling"]["profile"]["effort"]["class"] == "vendor",
          "mirai-s: the card's effort (xhigh, labelled vendor)",
          (mu["reasoning_effort"], mu["reasoning_budget_tokens"]))
    est = tiers.estimate_prompt_tokens(body)
    # THE THINKING BUDGET, LIKE THE OTHER MODELS (operator 2026-10-02: "use what we use scales for the tokens we
    # allow"; "Qwen overthinks so don't let it go forever"): bonsai's 20,480 x mirai-s's window 125,952 / 209,920
    check(mu["reasoning_budget_tokens"] == 20480 * 125952 // 209920 == 12288 and mu["max_tokens"] == 95232 - est
          and mu["_sampling"]["profile"]["thinking_cap"]["class"] == "operator",
          "mirai-s: thinking capped at bonsai's cap scaled by its window (12,288, the operator's rule); the turn is "
          "still the window - prompt", (mu["reasoning_budget_tokens"], mu["max_tokens"], est))
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
    check(wins["xhigh"] == 95232 and wins["max"] == 262144 and wins["medium"] == budget.budgets(model="bonsai")["main"],
          "the advertised window is per model: each tier's own", wins)
    card = catalog._chat_card(wins["medium"])
    tb = card["x_yamadori"]["reasoning_effort"].get("tokens_by_value") or {}
    check(tb.get("xhigh") == 95232 and tb.get("max") == 262144 and "mirai" not in json.dumps(card),
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
        cap = seen[-1]["cap"] or {}
        check(r.status_code == 200 and seen[-1]["model"] == "bonsai-a4000" and cap.get("model") == "bonsai-a4000",
              "server: high while xhigh holds the card -> served on bonsai-a4000 (operator 2026-10-06)", cap)
        lease.release()
        lease = max_mode.Lease(max_mode.decide("max", False))
        r = client.post("/v1/chat/completions", json={"model": "yamadori", "messages": msg,
                                                      "reasoning_effort": "xhigh"}, headers=h)
        body = r.json()
        check(r.status_code == 503 and body["error"]["code"] == "model_at_capacity"
              and body["error"].get("holder") == "flash-next" and body["error"].get("tier") == "xhigh",
              "server: xhigh (no other card) while max holds the card -> 503, the holder and the asked tier", body)
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


def test_decider_per_table_model():
    """JJAVA ON EVERY MODEL (operator 2026-09-29): a request mirai-s serves reads mirai-s's own decider profile
    (bench/decider/measure_model.py writes results/models/mirai-s.json) and keeps its label priors under its name."""
    reload(None, table=_table_file())
    import decider_bonsai as D
    import decide_turn
    tok, ctx = bound("mirai-s")
    try:
        path = D.profile_path()
        key = decide_turn._cf_key("x")
        name = D.canonical_model()
    finally:
        ctx.__exit__(None, None, None)
    check(name == "mirai-s" and path.replace("\\", "/").endswith("bench/decider/results/models/mirai-s.json")
          and key == "mirai-s:x",
          "a mirai-s request reads results/models/mirai-s.json and keys its label priors mirai-s:", (name, path, key))
    reload(None)


def test_locked_card_is_untouched():
    """FLASH-NEXT IS LOCKED (operator, 2026-09-30: "no jjava slowing down this highly tuned masterpiece, it stays locked
    in once it is swapped"): while it holds the card nothing but its own conversation touches it -- no decider, no
    side call, no release or slot read, no warm (the predicate the proxy's warm uses) -- and jjava and side calls go
    to bonsai-a4000. No network is reached anywhere below."""
    reload(None, table=_table_file())
    _LOADED.clear()
    _LOADED.add("flash-next")
    import decider_bonsai as D
    import model
    import vitals
    check(max_mode.locked("flash-next") and max_mode.locked("bonsai") and not max_mode.locked("mirai-s"),
          "the table: flash-next and bonsai locked (operator 2026-09-30); the fixture's mirai-s not (the unlocked path)")
    tok, ctx = bound("flash-next")
    try:
        dm = max_mode.decider_model()
        name = D.model_name()
    finally:
        ctx.__exit__(None, None, None)
    check(dm == "bonsai-a4000" and name == "bonsai-a4000",
          "a max request's jjava reads bonsai-a4000 (its profile and priors key there)", (dm, name))
    tok, ctx = bound("mirai-s")
    try:
        check(max_mode.decider_model() == "mirai-s", "mirai-s's jjava stays on its own card's lane until measured")
    finally:
        ctx.__exit__(None, None, None)
    d = max_mode.decide("medium", True)
    check(d.model == "bonsai-a4000" and not d.refuse, "a side call while flash-next holds the card -> bonsai-a4000",
          d.record())
    lease = max_mode.Lease(max_mode.decide("max", False))
    d2 = max_mode.decide("high", True)
    check(d2.model == "bonsai-a4000" and not d2.refuse, "... and while a max request is in flight, never refused",
          d2.record())
    lease.release()
    urls: list = []

    def fake_up(path, payload=None, timeout=30):
        raise AssertionError("not used")
    with NoNetwork() as nn:
        r = model.release_slot(0, model="flash-next")
        check(r.get("skipped") and "locked" in r["skipped"] and not nn.urls,
              "no release (and so no /slots read for one) touches the locked card", r)
        ok_w, why_w = max_mode.touch_allowed("warm", "flash-next")
        check(ok_w, "the conversation's own warm on the locked card IS allowed (coordinator 2026-09-30; "
              "proxy._warm_now's check)", why_w)
        ok_r, _ = max_mode.touch_allowed("slot_read", "flash-next")
        ok_s, _ = max_mode.touch_allowed("side_call", "flash-next")
        check(ok_r and not ok_s, "a /slots read is allowed; a side call (other work) is not")
    # THE DASHBOARD READS A LOCKED CARD'S /slots (coordinator 2026-09-30) -- at the loaded model's own port from
    # GET /running, never /upstream/<model>/ (which would load an unloaded one)
    seen: list = []
    saved_rows, saved_open2 = vitals.running_rows, urllib.request.urlopen

    class _R:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    def slots_only(req, *a, **k):
        url = req if isinstance(req, str) else req.full_url
        seen.append(url)
        if url.endswith(":5901/slots"):
            return _R(b'[{"id": 0, "is_processing": false, "n_ctx": 262144}]')
        raise AssertionError(f"unexpected read: {url}")
    vitals.running_rows = lambda max_age=0: [{"model": "flash-next", "state": "ready",
                                              "proxy": "http://127.0.0.1:5901"}]
    urllib.request.urlopen = slots_only
    try:
        v = vitals.slots()
    finally:
        vitals.running_rows, urllib.request.urlopen = saved_rows, saved_open2
    check(v.get("ok") and v.get("locked") and v.get("model") == "flash-next" and seen == [
        "http://127.0.0.1:5901/slots"] and not any("/upstream/" in u for u in seen),
          "the dashboard reads the locked card's /slots at its own port (flagged locked), never /upstream", (v, seen))
    seen.clear()
    vitals.running_rows = lambda max_age=0: []
    urllib.request.urlopen = slots_only
    try:
        vitals.slots()
    finally:
        vitals.running_rows, urllib.request.urlopen = saved_rows, saved_open2
    check(not any("/upstream/" in u for u in seen),
          "with nothing loaded no read goes through /upstream (the direct port only)", seen)
    with NoNetwork() as nn:
        # the decider's own door goes to bonsai-a4000's server, never flash-next's
        saved_open = urllib.request.urlopen

        def spy(req, *a, **k):
            urls.append(req if isinstance(req, str) else req.full_url)
            raise OSError("spy")
        urllib.request.urlopen = spy
        try:
            tok, ctx = bound("flash-next")
            try:
                try:
                    D._upstream("/tokenize", {"content": "A"})
                except OSError:
                    pass
            finally:
                ctx.__exit__(None, None, None)
        finally:
            urllib.request.urlopen = saved_open
    check(urls and all("/upstream/bonsai-a4000/" in u for u in urls),
          "the decider's reads go to bonsai-a4000, never the locked card", urls)
    released: list = []
    saved_rel = model.release_slot
    model.release_slot = lambda slot, **k: released.append(k.get("model")) or {"ok": True}
    try:
        tok, ctx = bound("flash-next")
        try:
            D.release(1, "decider turn")
        finally:
            ctx.__exit__(None, None, None)
    finally:
        model.release_slot = saved_rel
    check(released == ["bonsai-a4000"], "the decider's lane release goes to bonsai-a4000", released)
    _LOADED.clear()
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


def test_compaction_stays_on_its_card():
    """COMPACTIONS STAY ON THE CARD OF THE CONVERSATION THEY SUMMARISE (operator, 2026-09-30, verbatim: "it compacts
    on the same card it came from right? To get cache gains."): a compaction is not a side call for routing -- never
    bonsai-a4000, even at max, even while the card is locked; the one exception to "locked". And Bonsai's own jjava
    and side calls go to bonsai-a4000 ("2. Yes"), its main card -np 1."""
    reload(None, table=_table_file())
    _LOADED.clear()
    import tier_models
    t = tier_models.table()
    rows = {m: t.helpers(m) for m in t.models}
    check(all((h.get("decider") or "self") == (h.get("side_calls") or "self") for h in rows.values()),
          "every row names ONE helper for jjava and side calls (slots._helper_server_grant relies on it)", rows)
    check(rows["bonsai"].get("decider") == "bonsai-a4000" and rows["bonsai"].get("side_calls") == "bonsai-a4000",
          "bonsai's jjava and side calls: bonsai-a4000", rows["bonsai"])
    for card in ("bonsai", "flash-next"):
        _LOADED.clear()
        _LOADED.add(card)
        d = max_mode.decide("medium", True, kind="compaction")
        check(d.model == card and not d.refuse, f"a compaction while {card} holds the card stays on {card}",
              d.record())
        d = max_mode.decide("medium", True, kind="title")
        check(d.model == "bonsai-a4000", f"... while a title side call on {card}'s watch goes to bonsai-a4000",
              d.record())
        ok, why = max_mode.touch_allowed("compaction", card)
        check(ok, f"touch_allowed('compaction') on locked {card}: allowed (the one exception)", why)
        ok_d, _ = max_mode.touch_allowed("decider", card)
        check(not ok_d, f"... while jjava on locked {card} is not")
    _LOADED.clear()
    _LOADED.add("flash-next")
    lease = max_mode.Lease(max_mode.decide("max", False))
    d = max_mode.decide("max", True, kind="compaction")
    check(d.model == "flash-next" and not d.refuse, "a max conversation's compaction, its request in flight: flash-next",
          d.record())
    lease.release()
    _LOADED.clear()
    reload(None)


def test_internal_work_leaves_a_locked_card():
    """INTERNAL GENERATION NEVER REACHES A LOCKED CARD (coordinator, 2026-09-30): every generation through the one
    door (mcp/model.py post / chat / ask: summarize_text, the worker, the skill pipeline, skill_prove, the questions
    bank, skill_select's fallback decider) goes to the table's helper, bonsai-a4000, whenever the card's model is
    locked -- inside a conversation's request or not. An unlocked card keeps it; bonsai-vision is untouched. And
    none of those callers reaches a model except through that door."""
    import gpu_room
    import model
    reload(None, table=_table_file())
    sent: list = []

    class _R:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    def spy(req, *a, **k):
        url = req if isinstance(req, str) else req.full_url
        body = json.loads(req.data.decode()) if getattr(req, "data", None) else {}
        sent.append((url, body.get("model")))
        return _R(json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode())
    saved_open, saved_use = urllib.request.urlopen, gpu_room.use
    import contextlib
    gpu_room.use = lambda *a, **k: contextlib.nullcontext()
    urllib.request.urlopen = spy
    msgs = [{"role": "user", "content": "Summarize: a b c."}]
    try:
        for card in ("bonsai", "flash-next"):
            _LOADED.clear()
            _LOADED.add(card)
            sent.clear()
            model.ask(msgs)                                          # summarize_text, the worker, skill_select
            model.chat(msgs, effort="medium", max_tokens=64)         # the questions bank
            model.post(model.shape({"messages": msgs}, "medium", role="helper"))   # skill_pipeline / skill_prove
            tok, ctx = bound(card)
            try:
                model.ask(msgs)                                      # inside a conversation's request
            finally:
                ctx.__exit__(None, None, None)
            check(len(sent) == 4 and all(m == "bonsai-a4000" for _, m in sent)
                  and not any("/upstream/" in u for u, _ in sent),
                  f"{card} locked on the card: every internal generation goes to bonsai-a4000, none to the card",
                  sent)
        _LOADED.clear()
        _LOADED.add("mirai-s")
        sent.clear()
        tok, ctx = bound("mirai-s")
        try:
            model.ask(msgs)
        finally:
            ctx.__exit__(None, None, None)
        check([m for _, m in sent] == ["mirai-s"], "an UNLOCKED card (mirai-s) keeps its own internal work", sent)
        _LOADED.clear()
        _LOADED.add("bonsai")
        check(max_mode.internal_model("bonsai-vision") == "bonsai-vision"
              and max_mode.internal_model("bonsai-a4000") == "bonsai-a4000",
              "bonsai-vision and the helper itself are not re-routed")
    finally:
        urllib.request.urlopen, gpu_room.use = saved_open, saved_use
        _LOADED.clear()
        reload(None)
    here = os.path.dirname(os.path.abspath(__file__))
    direct = []
    for f in ("code_search.py", "worker.py", "skill_pipeline.py", "skill_prove.py", "skill_questions_bank.py",
              "skill_select.py", "skill_deciders.py"):
        src = open(os.path.join(here, f), encoding="utf-8").read()
        if "chat/completions" in src or re.search(r"[\"'/]completion[\"'?]", src):
            direct.append(f)
    check(not direct, "none of those callers generates except through mcp/model.py (the one door)", direct)


def test_helper_window_and_fit():
    """A JOB ROUTED TO bonsai-a4000 IS BUDGETED AGAINST ITS OWN WINDOW (coordinator, 2026-09-30): the table's fitted
    141,312 (main cap 138,240), not the main line; an oversize job is refused with its sizes BEFORE anything is sent
    (model.WindowExceeded), never cut off mid-generation; and a job claimed for the A4000 never reaches the main card
    (max_mode.check_scope)."""
    import budget
    import model
    import tiers
    reload(None, table=_table_file())
    _LOADED.clear()
    _LOADED.add("bonsai")
    saved_open = urllib.request.urlopen
    try:
        w = budget.model_window("bonsai-a4000") or {}
        check(w.get("ctx") == 141312 and w.get("main_cap") == 138240,
              "the table carries bonsai-a4000's fitted window (bench/a4000_fit.py)", w)
        check(tiers.window_model_of("bonsai-a4000") == "bonsai-a4000" and tiers.window_model_of("bonsai") is None
              and tiers.window_model_of("yamadori") is None,
              "only a helper server's own window replaces the request's (a main model's follows it as before)")
        big = "word " * 60000                                         # ~100K tokens by the high estimate
        shaped = model.shape({"messages": [{"role": "user", "content": big}]}, "medium", role="helper")
        est = tiers.estimate_prompt_tokens(shaped)
        check(shaped.get("model") == "bonsai-a4000" and shaped["max_tokens"] + est <= 138240,
              "a locked card's job is shaped for bonsai-a4000: prompt + max_tokens within its 138,240 cap",
              {"model": shaped.get("model"), "max_tokens": shaped.get("max_tokens"), "prompt_est": est})
        sent: list = []
        urllib.request.urlopen = lambda req, *a, **k: sent.append(req) or (_ for _ in ()).throw(OSError("x"))
        huge = "word " * 90000                                        # ~150K tokens by the estimate: past 138,240
        try:
            model.post(model.shape({"messages": [{"role": "user", "content": huge}]}, "medium"))
            check(False, "an oversize job is refused")
        except model.WindowExceeded as e:
            check(e.model == "bonsai-a4000" and e.window == 138240 and e.prompt > 138240 - e.floor and not sent
                  and "retryable: no" in str(e),
                  "an oversize job is refused before anything is sent, with its sizes (prompt, floor, window)",
                  str(e)[:300])
        send = {"model": "bonsai-a4000", "messages": [{"role": "user", "content": big}], "max_tokens": 200000,
                "reasoning_budget_tokens": 190000}
        rec = model.fit_window(send)
        check(rec and send["max_tokens"] + tiers.estimate_prompt_tokens(send) <= 138240
              and send["reasoning_budget_tokens"] <= send["max_tokens"] - tiers.A_MIN,
              "a request that fits has max_tokens and its thinking cut to what the window leaves", rec)
        check(model.fit_window({"model": "bonsai", "messages": [], "max_tokens": 10**6}) is None,
              "the main card's own requests are untouched by the helper's window")
        # the card scope
        _LOADED.clear()
        _LOADED.add("mirai-s")
        max_mode.set_scope("gpu_a4000")
        try:
            try:
                max_mode.check_scope("mirai-s")
                check(False, "a job claimed for the A4000 cannot reach the main card")
            except max_mode.ModelAtCapacity as e:
                check("gpu_a4000" in str(e), "a job claimed for the A4000 whose route names a main model waits "
                      "(ModelAtCapacity: the worker defers it)", str(e)[:200])
            max_mode.check_scope("bonsai-a4000")
            check(True, "... and reaches bonsai-a4000 freely")
        finally:
            max_mode.set_scope(None)
        max_mode.check_scope("mirai-s")
        check(True, "outside an A4000 job nothing changes")
    finally:
        urllib.request.urlopen = saved_open
        _LOADED.clear()
        reload(None)


def test_gpu_lane_per_card():
    """THE WORKER'S GPU LANE PER CARD (coordinator, 2026-09-30; mcp/jobs.py GPU_SCOPES): an embedding job is the
    A4000's always; a model stage's job is the A4000's while the main card is locked, the main card's otherwise; a
    5060 Ti window's jobs.pause("gpu") no longer holds the A4000's jobs (and pause("gpu_a4000") holds only them);
    one running job per card; anything unrouted keeps lane gpu."""
    import jobs
    reload(None, table=_table_file())
    try:
        _LOADED.clear()
        _LOADED.add("bonsai")
        check(jobs.internal_off_main(), "bonsai locked on the card: internal generation is off the main card")
        e = jobs.add("dataset.index", {}, lane="gpu")               # a legacy row: queued on lane gpu
        m = jobs.add("skill.prove", {}, lane="gpu")
        u = jobs.add("test.unrouted", {}, lane="gpu")
        jobs.pause("gpu", by="test", why="a 5060 Ti window")
        got = [jobs.claim("gpu_a4000", "t:a")]
        check(got[0] and got[0]["id"] in (e, m) and got[0]["card"] == "gpu_a4000",
              "the 5060 Ti's pause does not hold the A4000's jobs", got)
        check(jobs.claim("gpu_a4000", "t:a2") is None, "one running job per card (the A4000's slot is taken)")
        check(jobs.claim("gpu", "t:g") is None, "the paused main card hands out nothing")
        jobs.finish(got[0]["id"], {})
        second = jobs.claim("gpu_a4000", "t:a")
        check(second and {got[0]["id"], second["id"]} == {e, m},
              "the embedding job and the locked card's model job both run on the A4000", second)
        jobs.resume("gpu")
        g = jobs.claim("gpu", "t:g")
        check(g and g["id"] == u and g["card"] == "gpu", "an unrouted job stays on the main card's lane", g)
        jobs.finish(second["id"], {})
        jobs.finish(g["id"], {})
        # unlocked card: a model job is the main card's
        _LOADED.clear()
        _LOADED.add("mirai-s")
        m2 = jobs.add("skill.review", {}, lane="gpu")
        check(jobs.claim("gpu_a4000", "t:a") is None, "an unlocked card (mirai-s): a model job is not the A4000's")
        g2 = jobs.claim("gpu", "t:g")
        check(g2 and g2["id"] == m2, "... it runs on the main card's lane", g2)
        jobs.finish(m2, {})
        jobs.pause("gpu_a4000", by="test", why="an A4000 window")
        e2 = jobs.add("package.knn", {}, lane="gpu_a4000")
        check(jobs.claim("gpu_a4000", "t:a") is None and jobs.claimable("gpu_a4000") == 1,
              "an A4000 window pauses only its own card's jobs", e2)
        jobs.resume("gpu_a4000")
        jobs.finish(jobs.claim("gpu_a4000", "t:a")["id"], {})
    finally:
        jobs.resume("gpu")
        jobs.resume("gpu_a4000")
        _LOADED.clear()
        reload(None)


def test_other_card_table():
    """THE OTHER CARD in the table (operator, 2026-09-30): bonsai's other card is bonsai-a4000; flash-next and mirai-s
    have none (their newcomer is refused, never downgraded) unless the one switch YAMADORI_OTHER_CARD_DOWNGRADE=1."""
    reload(None, table=_table_file())
    try:
        oc, _ = max_mode.other_card("bonsai")
        fn, why_fn = max_mode.other_card("flash-next")
        ms, _ = max_mode.other_card("mirai-s")
        check(oc == "bonsai-a4000" and fn is None and ms is None and "does not run" in why_fn,
              "bonsai -> bonsai-a4000; flash-next and mirai-s: none", (oc, fn, ms, why_fn))
        os.environ["YAMADORI_OTHER_CARD_DOWNGRADE"] = "1"
        dn, why_dn = max_mode.other_card("flash-next")
        check(dn == "bonsai-a4000" and "DOWNGRADED" in why_dn,
              "the one switch YAMADORI_OTHER_CARD_DOWNGRADE=1 serves them on the default model's, and says so", why_dn)
    finally:
        os.environ.pop("YAMADORI_OTHER_CARD_DOWNGRADE", None)
        reload(None)
    check(max_mode.other_card("bonsai")[0] is None, "with the table off: no other card (the 503 as before)")


def test_rerouted_turn_releases_its_lease():
    """A CONVERSATION ROUTED TO THE OTHER CARD HOLDS NO MAIN-CARD LEASE (coordinator, 2026-09-30): server._serve_turn
    took a lease for bonsai; once proxy._run_turn re-binds the turn to bonsai-a4000 (max_mode.set_current, after
    slots.check_owner routed it), the lease is released -- so a max request's swap to flash-next is not held behind
    it. A main -> main re-bind and a side call's own helper lease are untouched; the server's later release is a
    no-op. The blocking path carries the lease on its token too (server._complete_on)."""
    import server
    reload(None, table=_table_file())
    _LOADED.clear()
    _LOADED.add("bonsai")
    try:
        lease = max_mode.Lease(max_mode.decide("medium", False))
        check(lease.model == "bonsai" and max_mode._inflight.get("bonsai") == 1, "a bonsai turn holds its lease")
        tok = cancel.Token()
        tok.yamadori_lease = lease
        with cancel.bound(tok):
            max_mode.set_current("bonsai")                       # the decision's own model: nothing changes
            held = not lease.released
            max_mode.set_current("bonsai-a4000")                 # routed to the other card
        check(held and lease.released and max_mode._inflight.get("bonsai") == 0
              and (getattr(tok, "yamadori_rerouted", None) or {}).get("to") == "bonsai-a4000",
              "routed to bonsai-a4000: the bonsai lease is released (nothing in flight on the main card)",
              getattr(tok, "yamadori_rerouted", None))
        mx = max_mode.decide("max", False)
        check(mx.model == "flash-next" and not mx.refuse and max_mode._switching_to in (None, "flash-next"),
              "a max request then finds no bonsai work in flight to wait behind", mx.record())
        max_mode.release(lease)
        check(max_mode._inflight.get("bonsai") == 0, "the server's later release is a no-op (never below zero)")
        side = max_mode.Lease(max_mode.decide("medium", True))
        tok2 = cancel.Token()
        tok2.yamadori_lease = side
        with cancel.bound(tok2):
            max_mode.set_current(side.model)
        check(side.model == "bonsai-a4000" and not side.released,
              "a side call's own bonsai-a4000 lease is untouched")
        side.release()
        # the blocking path: the lease rides on the token proxy.complete runs under
        seen = {}
        saved = server.proxy.complete

        def fake_complete(body):
            seen["lease"] = getattr(cancel.current(), "yamadori_lease", None)
            return {}
        server.proxy.complete = fake_complete
        try:
            t = cancel.Token()
            t.yamadori_lease = "L"
            server._complete_on(t, {})
        finally:
            server.proxy.complete = saved
        check(seen.get("lease") == "L", "the blocking path runs the turn under the token carrying its lease")
    finally:
        _LOADED.clear()
        reload(None)


def test_utility_kind():
    """max_mode.utility_kind: None for a task turn, 'compaction' for a harness's summary request (the kind the
    routing above reads; selection.utility_kind)."""
    reload(None, table=_table_file())
    task = {"messages": [{"role": "user", "content": "Write a function that adds two numbers."}]}
    comp = {"messages": [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": "Summarize this conversation so far for a continuation."}]}
    k0, k1 = max_mode.utility_kind(task), max_mode.utility_kind(comp)
    check(k0 is None and max_mode.is_utility(task) is False, "a task turn has no utility kind", k0)
    check(k1 == "compaction" and max_mode.is_utility(comp), "a summarise request is a compaction", k1)
    reload(None)


def main() -> int:
    for fn in (test_off_by_default, test_routing, test_capacity_refusal, test_swap_waits_never_cancels,
               test_inheritance_and_no_reload, test_worker_defers, test_decider_per_model,
               test_thinking_at_max_follows_the_card, test_server_admission,
               test_table_routing_and_ranks, test_table_capacity_and_preemption, test_table_profile_applied,
               test_table_windows_and_vision, test_table_server_records, test_no_offline_swap,
               test_legacy_is_inert, test_decider_per_table_model, test_locked_card_is_untouched,
               test_compaction_stays_on_its_card, test_utility_kind, test_internal_work_leaves_a_locked_card,
               test_helper_window_and_fit, test_gpu_lane_per_card, test_other_card_table,
               test_rerouted_turn_releases_its_lease, test_live_table_locks_every_main_model):
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
