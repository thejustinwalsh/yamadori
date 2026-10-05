#!/usr/bin/env python
"""THE DASHBOARD SHOWS WHAT RUNS WHERE, and follows the model that is ACTUALLY on the card (offline).

The operator, 2026-10-05: "the dashboard is not showing me the other agent lanes at all, some weird line about
someone else holding the GPU, or something. I want to know the status of anything running in the bargraph, we should
be swapping out the loadouts if that is what we need, Flash-Next is not showing anything when running in the
dashboard."

WHAT IS GATED (mcp/vitals.py, mcp/lane_view.py, mcp/power.py, mcp/max_mode.py, mcp/slots.py, mcp/gpu_room.py)

  1. FOLLOWS THE LOADED MODEL: with flash-next, mirai-s or bonsai ready in llama-swap's /running, /slots is read from
     THAT model's own port, as 127.0.0.1 (llama-swap says `localhost`: IPv6 first, ~1 s of the 1 s timeout -- live,
     2026-10-05: a ready flash-next /slots answered in 1,031 ms), per-model rates, serving.on_card names it; the
     context is its table window (flash-next 262,144, read from nothing), the default model's from its own port.
  2. THE LINE: nothing loaded / a swap under way / a model starting are said in plain words; a paused job lane is
     said as a pause on the worker's queue and never as the card's holder.
  3. THE LANE VIEW: every card, the models on it with their own slots and what each slot is for, the A4000's
     expected-but-unloaded helper, the worker's job lanes (paused, running, waiting), each hold with its reason and
     since when, the swap in progress and the last swaps.
  4. KOGOSEI (power.SlotCounter) follows the card's model across a swap.
  5. max_mode records the swap in progress (`swap_now`) while wait_ready loads, and the age of each lease.

NO NETWORK: urllib.request.urlopen is a table of fakes; a URL not in it fails the suite.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_dash_lanes_")
os.environ["YAMADORI_MAX_STATE"] = os.path.join(_TMP, "max_mode.json")
os.environ["YAMADORI_TIER_MODELS"] = os.path.join(HERE, "tier_models.yaml")
os.environ["YAMADORI_GPU_ROOM"] = "0"
os.environ["CONCEPT_SEED_LAST"] = os.path.join(_TMP, "seed_last.json")
os.makedirs(os.environ["YAMADORI_GPU_ROOM_DIR"], exist_ok=True)

import budget  # noqa: E402
import gpu_room  # noqa: E402
import jobs  # noqa: E402
import lane_view  # noqa: E402
import max_mode  # noqa: E402
import power  # noqa: E402
import slots as slot_map  # noqa: E402
import vitals  # noqa: E402

_results: list[tuple[bool, str, str]] = []
_leaks: list[str] = []
_asked: list[str] = []


def check(ok, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, str(detail)[:500]))
    return bool(ok)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# url -> the JSON body the fake server answers
BODIES: dict[str, object] = {}


def _urlopen(url, timeout=None):                                  # noqa: ARG001
    u = url if isinstance(url, str) else url.full_url
    _asked.append(u)
    if u in BODIES:
        b = BODIES[u]
        if isinstance(b, Exception):
            raise b
        return _Resp(json.dumps(b).encode())
    _leaks.append(u)
    raise OSError(f"test_dash_lanes: {u} is not a fake")


urllib.request.urlopen = _urlopen
vitals._sh = lambda cmd, timeout=25: (                            # noqa: E731
    "0, NVIDIA GeForce RTX 5060 Ti, 15318, 16311, 35, GPU-de660e90-0e9c-d465-b389-6df63021b920, 60, 180, 40\n"
    "1, NVIDIA RTX A4000, 1969, 16376, 0, GPU-43e37d0c-4104-9056-2552-6109d4d3382c, 5, 140, 30\n" if cmd and cmd[0] == "nvidia-smi" else "")
vitals._sampled_gpus = lambda max_age: None
# what max_mode believes is loaded: the same /running rows the test sets (its own read would go to llama-swap)
max_mode.running = lambda: {r["model"] for r in (vitals._running_cache[1] or []) if r["state"] != "stopped"}

FLASH = "http://localhost:10008"
BONSAI = "http://localhost:10001"
MIRAI = "http://localhost:10003"
A4000 = "http://localhost:10006"
EMB = "http://localhost:10007"

CMD_FLASH = "llama-server -m /m/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --port 10008"
CMD_BONSAI = "llama-server -m /m/Ternary-Bonsai-2-27B.gguf --port 10001"
CMD_A4000 = "llama-server -m /m/Ternary-Bonsai-2-27B.gguf --port 10006 -np 2"
CMD_EMB = "llama-server -m /m/Qwen3-Embedding-0.6B-Q8_0.gguf --port 10007 --embeddings"


def row(model, state, proxy, cmd, ttl=0):
    return {"model": model, "state": state, "proxy": proxy, "cmd": cmd, "ttl": ttl}


def running(*rows):
    vitals._running_cache = (time.time(), list(rows) if rows is not None else None)
    max_mode._running_cache = (time.time(), {r["model"] for r in rows if r["state"] != "stopped"})
    lane_view._first_seen.clear()


def slot_json(n, ctx=262144, busy_ids=(), decoded=100):
    out = []
    for i in range(n):
        busy = i in busy_ids
        out.append({"id": i, "n_ctx": ctx, "is_processing": busy, "id_task": 40 + i,
                    "n_prompt_tokens": 5000 + decoded if busy else 0,
                    "n_prompt_tokens_processed": 5000 if busy else 0,
                    "next_token": [{"n_decoded": decoded if busy else 0, "n_remain": 1000}]})
    return out


def reset() -> None:
    BODIES.clear()
    _asked.clear()
    vitals._slot_prev.clear()
    vitals._slot_rate.clear()
    running()
    budget._POOL = 209920
    max_mode._reset_for_tests()
    for lane in jobs.GPU_SCOPES:
        jobs.resume(lane)


# ---------------------------------------------------------------- 1. follows

def test_slots_follow_the_loaded_model():
    check(max_mode.ENABLED and max_mode.MODELS == ["bonsai", "mirai-s", "flash-next"],
          "the live table is in force: three main models", max_mode.MODELS)
    for model, proxy, cmd, port in (("flash-next", FLASH, CMD_FLASH, 10008), ("mirai-s", MIRAI, CMD_BONSAI, 10003),
                                    ("bonsai", BONSAI, CMD_BONSAI, 10001)):
        reset()
        running(row(model, "ready", proxy, cmd), row("embeddings", "ready", EMB, CMD_EMB))
        BODIES[f"http://127.0.0.1:{port}/slots"] = slot_json(1, busy_ids=(0,))
        url, m = vitals._slots_target()
        check(url == f"http://127.0.0.1:{port}/slots" and m == model,
              f"{model} on the card: /slots is read from ITS port, as 127.0.0.1 (never localhost)", (url, m))
        r = vitals.slots()
        check(r["ok"] and r["model"] == model and r["decoding"] == 1 and not any("localhost" in u for u in _asked),
              f"{model}: the read answers, names the model, and never asked localhost", (r, _asked))
        sv = vitals.serving()
        check(sv["on_card"] == model, f"serving.on_card is the model that is loaded ({model}), not an assumed one", sv)
    reset()
    running(row("flash-next", "ready", FLASH, CMD_FLASH))
    BODIES["http://127.0.0.1:10008/slots"] = slot_json(1, busy_ids=(0,), decoded=100)
    vitals.slots()
    check(("flash-next", 0) in vitals._slot_prev, "rates are kept per (model, slot), not per slot number",
          list(vitals._slot_prev))
    BODIES["http://127.0.0.1:10008/slots"] = slot_json(1, busy_ids=(0,), decoded=100)
    BODIES["http://127.0.0.1:10001/slots"] = slot_json(1, busy_ids=(0,), decoded=7)
    check(vitals._slots_target()[1] == "flash-next", "flash-next still the card's model")


def test_a_model_that_is_starting_is_not_read():
    reset()
    running(row("flash-next", "starting", FLASH, CMD_FLASH))
    BODIES["http://127.0.0.1:10001/slots"] = OSError("refused")
    url, m = vitals._slots_target()
    check(url == vitals.SLOTS_URL and m == vitals.MAIN_MODEL,
          "a model still starting is not asked for /slots (it would hold the read until the timeout)", (url, m))
    r = vitals.slots()
    check(r["ok"] is False and "flash-next is starting" in r["error"] and r["on_card"] == "flash-next",
          "and the failure says what it is: flash-next is starting", r)


def test_context_follows_the_loaded_model():
    reset()
    running(row("flash-next", "ready", FLASH, CMD_FLASH))
    c = vitals.context_pool()
    check(c.get("pool") == 262144 and c.get("model") == "flash-next" and "tier table" in str(c.get("pool_read"))
          and not any(u.endswith("/props") for u in _asked),
          "flash-next on the card: its table window (262,144), read from nothing, never bonsai's pool", (c, _asked))
    reset()
    running(row("bonsai", "ready", BONSAI, CMD_BONSAI))
    BODIES["http://127.0.0.1:10001/props"] = {"default_generation_settings": {"n_ctx": 209920}, "total_slots": 1,
                                             "kv_vram_cells": 209920}
    c = vitals.context_pool()
    check(c.get("pool") == 209920 and c.get("model") == "bonsai"
          and _asked == ["http://127.0.0.1:10001/props"] and "10001/props" in str(c.get("pool_read")),
          "bonsai on the card: the pool is read from the port /running reports for it", (c, _asked))
    reset()
    running(row("embeddings", "ready", EMB, CMD_EMB))
    c = vitals.context_pool()
    check("not loaded" in str(c.get("pool_read") or c.get("error")) and not _asked,
          "no main model loaded: nothing is asked, and the pool says why", (c, _asked))
    reset()
    running(row("mirai-s", "ready", MIRAI, CMD_BONSAI))
    check(vitals.cached_context(model="mirai-s").get("model") == "mirai-s",
          "the pulse's cheap context follows the card's model too (mirai-s)")


# ------------------------------------------------------------------- 2. line

def test_the_line_says_what_holds_what():
    reset()
    jobs.pause("gpu", "coordinator", "operator sessions 2026-10-02: held until the skills window", 7200)
    running(row("embeddings", "ready", EMB, CMD_EMB))
    why = vitals.off_card_why()
    check("no main model is loaded" in why and "paused by coordinator" in why and "not held by it" in why
          and "holds it" not in why and "operator sessions" not in why,
          "nothing loaded + a paused lane: two plain facts, the lane never named the card's holder", why)
    jobs.resume("gpu")
    running(row("bonsai", "ready", BONSAI, CMD_BONSAI), row("flash-next", "starting", FLASH, CMD_FLASH))
    max_mode._swap_now = {"to": "flash-next", "from": ["bonsai"], "phase": "loading", "since": time.time() - 12}
    why = vitals.off_card_why()
    check(why.startswith("swapping the main card: bonsai -> flash-next: loading it") and "12 s so far" in why,
          "a swap under way: from -> to, the phase and the seconds so far", why)
    max_mode._swap_now = {"to": "flash-next", "from": ["bonsai"], "phase": "waiting", "since": time.time() - 3}
    check("waiting for the work in flight on bonsai to end" in vitals.off_card_why(),
          "waiting for another model's work to end is its own phase", vitals.off_card_why())
    max_mode._swap_now = None
    running(row("flash-next", "ready", FLASH, CMD_FLASH))
    check(vitals.off_card_why() is None, "a ready model on the card: no reason to give (the read's own error stands)")


# ----------------------------------------------------------------- 3. lanes

def _two_cards():
    running(row("flash-next", "ready", FLASH, CMD_FLASH),
            row("embeddings", "ready", EMB, CMD_EMB),
            row("bonsai-a4000", "ready", A4000, CMD_A4000, ttl=0))
    BODIES["http://127.0.0.1:10008/slots"] = slot_json(1, busy_ids=(0,), decoded=250)
    BODIES["http://127.0.0.1:10006/slots"] = slot_json(2, ctx=138240, busy_ids=(1,), decoded=5)


def test_cards_models_and_slots():
    reset()
    _two_cards()
    v = lane_view.snapshot()
    by = {c["key"]: c for c in v["cards"]}
    check(set(by) == {"main", "a4000"} and by["main"]["name"].endswith("5060 Ti") and by["a4000"]["name"].endswith("A4000"),
          "two cards: the 5060 Ti is main, the A4000 is a4000", [(c["key"], c["name"]) for c in v["cards"]])
    mm = {m["model"]: m for m in by["main"]["models"]}
    am = {m["model"]: m for m in by["a4000"]["models"]}
    check(list(mm) == ["flash-next"] and mm["flash-next"]["port"] == 10008 and mm["flash-next"]["slots"]["ok"]
          and mm["flash-next"]["slots"]["slots"][0]["state"] == "decode"
          and mm["flash-next"]["slots"]["slots"][0]["use"] == "conversation",
          "the main card: flash-next with ITS slots (decoding), the slot's use named", mm)
    s = am["bonsai-a4000"]["slots"]["slots"]
    check([x["use"] for x in s] == ["second conversation (the other card)", "jjava, side calls"]
          and s[0]["state"] == "idle" and s[1]["state"] in ("decode", "prefill"),
          "the A4000's bonsai-a4000: slot 0 the other card's conversation, slot 1 jjava and side calls", s)
    check(am["embeddings"]["slots"] is None and "embeddings" in am,
          "the embedder is listed on the A4000 and is not asked for /slots (it has none)", am["embeddings"])
    check(by["a4000"]["expected"] == [], "bonsai-a4000 is loaded: nothing expected is missing")
    check(not any("localhost" in u for u in _asked), "every /slots was read at 127.0.0.1", _asked)
    reset()
    running(row("flash-next", "ready", FLASH, CMD_FLASH), row("embeddings", "ready", EMB, CMD_EMB))
    BODIES["http://127.0.0.1:10008/slots"] = slot_json(1)
    v = lane_view.snapshot()
    ex = {c["key"]: c for c in v["cards"]}["a4000"]["expected"]
    check(len(ex) == 1 and ex[0]["model"] == "bonsai-a4000" and "flash-next" in ex[0]["for"],
          "flash-next is locked and bonsai-a4000 is not loaded: the A4000 says it is expected, and why", ex)
    reset()
    running(row("flash-next", "ready", FLASH, CMD_FLASH))
    BODIES["http://127.0.0.1:10008/slots"] = OSError("timed out")
    v = lane_view.snapshot()
    m = v["cards"][0]["models"][0]
    check(m["slots"]["ok"] is False and "timed out" in m["slots"]["error"],
          "a /slots read that fails says so on that model alone", m["slots"])


def test_holds_in_plain_words():
    reset()
    _two_cards()
    jobs.pause("gpu", "coordinator", "the skills window", 3600)
    jobs.pause("gpu_a4000", "coordinator", "the skills window", 3600)
    # a running job on the A4000's scope, and one waiting
    a = jobs.add("dataset.index", {"x": 1}, lane="gpu")
    jobs.resume("gpu_a4000")
    got = jobs.claim("gpu_a4000")
    jobs.pause("gpu_a4000", "coordinator", "the skills window", 3600)
    jobs.add("skill.match_index", {}, lane="gpu")
    now = time.time()
    with max_mode._lock:
        max_mode._inflight["flash-next"] = 2
        max_mode._since["flash-next"] = now - 41
    real_h, real_l, real_en = slot_map.holds, gpu_room.leases, gpu_room.enabled
    slot_map.holds = lambda: {"hold_s": 60.0, "one_conversation": True, "other_model": "bonsai-a4000",
                              "main": {"owner": "ab12cd34", "busy": False, "idle_s": 20.0, "hold_s": 60.0, "rest_s": 41},
                              "other": {"owner": "ee99ff00", "busy": True, "idle_s": 0.0, "hold_s": 60.0, "rest_s": 30}}
    gpu_room.enabled = lambda: True
    gpu_room.leases = lambda: [{"model": "bonsai-vision", "pid": 4242, "since": now - 9, "until": now + 3000}]
    max_mode._swap_now = {"to": "mirai-s", "from": ["flash-next"], "phase": "waiting", "since": now - 5}
    try:
        v = lane_view.snapshot()
    finally:
        slot_map.holds, gpu_room.leases, gpu_room.enabled = real_h, real_l, real_en
    kinds = [(h["kind"], h["card"]) for h in v["holds"]]
    for k in (("job_lane_paused", "main"), ("job_lane_paused", "a4000"), ("model_lease", "main"),
              ("conversation", "main"), ("conversation", "a4000"), ("gpu_room_lease", "a4000"), ("swap", "main")):
        check(k in kinds, f"hold listed: {k[0]} on {k[1]}", kinds)
    hb = {(h["kind"], h["card"]): h for h in v["holds"]}
    p = hb[("job_lane_paused", "main")]
    check(p["by"] == "coordinator" and p["why"] == "the skills window" and p["since"] and p["until"] > now
          and "not a hold on the card" in p["what"], "a paused lane: by, why, since, until, and said as a pause", p)
    l = hb[("model_lease", "main")]
    check("2 requests in flight on flash-next" in l["what"] and abs(l["since"] - (now - 41)) < 1,
          "a lease: how many requests, on which model, since when", l)
    c = hb[("conversation", "main")]
    check("ab12cd34 holds the main card for 41 more s" in c["what"] and "goes to the other card" in c["what"],
          "a conversation's hold: whose, for how much longer, what a newcomer gets", c)
    check("has a request in flight on the other card" in hb[("conversation", "a4000")]["what"],
          "the other card's conversation with a request in flight is said so")
    check("waiting for the work in flight on flash-next to end before loading mirai-s" in hb[("swap", "main")]["what"]
          and hb[("swap", "main")]["since"] == now - 5, "a swap in progress is a hold with its phase and since")
    jl = {j["lane"]: j for j in v["jobs"]}
    check(jl["gpu_a4000"]["paused"] and len(jl["gpu_a4000"]["running"]) == 1
          and jl["gpu_a4000"]["running"][0]["queue"] == "dataset.index" and jl["gpu_a4000"]["queued"] == 1,
          "the A4000's job lane: paused, one job running (which), one waiting", jl["gpu_a4000"])
    check(jl["gpu"]["paused"] and jl["gpu"]["running"] == [] and jl["gpu"]["queued"] == 0,
          "the main card's lane: paused, nothing running", jl["gpu"])
    check(got is not None and a, "(the fixture claimed a job for the A4000's scope)")
    max_mode._swap_now = None


def test_swaps_view():
    reset()
    running(row("flash-next", "ready", FLASH, CMD_FLASH))
    BODIES["http://127.0.0.1:10008/slots"] = slot_json(1)
    import stats_store
    real = stats_store.read
    t = time.time()
    stats_store.read = lambda table, since, *a, **k: [
        {"ts": t - 900, "from_models": '["mirai-s"]', "to_model": "bonsai", "load_s": 8.6, "ok": 1, "how": "HTTP 200",
         "left_loaded": "[]"},
        {"ts": t - 300, "from_models": '["bonsai"]', "to_model": "flash-next", "load_s": 20.1, "ok": 1, "how": "HTTP 200",
         "left_loaded": "[]"}] if table == "swaps" else []
    try:
        v = lane_view.snapshot()
    finally:
        stats_store.read = real
    last = v["swap"]["last"]
    check([x["to"] for x in last] == ["flash-next", "bonsai"] and last[0]["from"] == ["bonsai"] and last[0]["load_s"] == 20.1,
          "the last swaps, newest first: from -> to and the load seconds", last)
    check(v["swap"]["now"] is None, "no swap in progress")
    running(row("bonsai", "starting", BONSAI, CMD_BONSAI))
    v = lane_view.snapshot()
    check(any(h["kind"] == "model_state" and "bonsai is starting" in h["what"] and h["since"] for h in v["holds"]),
          "a model llama-swap is starting is listed, since when this view first saw it",
          [h for h in v["holds"]])


def test_pulse_carries_the_view():
    reset()
    _two_cards()
    p = vitals.pulse()
    check(isinstance(p.get("cards"), dict) and p["cards"].get("cards") and p["slots"]["model"] == "flash-next",
          "the pulse carries `cards` beside `slots`", list(p))
    check(p["context"].get("model") == "flash-next" and p["context"]["pool"] == 262144,
          "and its context is the loaded model's, not bonsai's", p["context"])


# -------------------------------------------------------------- 4. kogosei

def test_the_power_reader_follows_the_card():
    calls: list[str] = []
    state = {"running": {"running": [{"model": "bonsai", "state": "ready", "proxy": "http://localhost:10001"}]},
             "slots": [{"id": 0, "id_task": 1, "is_processing": True, "n_prompt_tokens_processed": 100,
                        "next_token": [{"n_decoded": 10}]}]}

    def get(url, timeout):                                        # noqa: ARG001
        calls.append(url)
        return state["slots"] if url.endswith("/slots") else state["running"]
    t = iter(range(1000, 2000))
    c = power.SlotCounter(get=get, clock=lambda: next(t))
    r = c.read()
    check(r["state"] == "ok" and r["model"] == "bonsai", "bonsai on the card: read, and named", r)
    state["running"] = {"running": [{"model": "flash-next", "state": "ready", "proxy": "http://localhost:10008"}]}
    c._url_at = -1e9                          # the 30 s url cache has lapsed
    calls.clear()
    r = c.read()
    check(r["state"] == "ok" and r["model"] == "flash-next" and r["decoded"] is None
          and "http://127.0.0.1:10008/slots" in calls,
          "after a swap the reader follows flash-next: its own port, its counters primed afresh", (r, calls))
    state["slots"] = [{"id": 0, "id_task": 1, "is_processing": True, "n_prompt_tokens_processed": 100,
                       "next_token": [{"n_decoded": 40}]}]
    r = c.read()
    check(r["decoded"] == 30 and r["model"] == "flash-next", "and then counts its tokens", r)
    state["running"] = {"running": [{"model": "embeddings", "state": "ready", "proxy": "http://localhost:10007"}]}
    c._url_at = -1e9
    check(c.read()["state"] == "not_loaded", "no main model loaded: not_loaded")
    pinned = power.SlotCounter(get=get, clock=lambda: next(t), model="bonsai")
    state["running"] = {"running": [{"model": "flash-next", "state": "ready", "proxy": "http://localhost:10008"}]}
    check(pinned.read()["state"] == "not_loaded", "a reader pinned to a model reads only that model")


# -------------------------------------------------------------- 5. max_mode

def test_max_mode_records_the_swap_in_progress():
    reset()
    seen: dict = {}

    def fake_load(model, timeout=0):                              # noqa: ARG001
        seen["during"] = max_mode.snapshot()["swap_now"]
        return True, "HTTP 200"
    real = max_mode._load
    max_mode._load = fake_load
    lease = max_mode.Lease(max_mode.decide("max", False))
    try:
        check(max_mode.snapshot()["inflight_since"].get("flash-next", 0) > 0,
              "a lease records when the model's oldest request in flight took it")
        out = max_mode.wait_ready("flash-next")
    finally:
        max_mode._load = real
        lease.release()
    d = seen.get("during") or {}
    check(d.get("to") == "flash-next" and d.get("phase") == "loading" and d.get("since"),
          "while the model loads, swap_now says to, the phase and since", seen)
    check(max_mode.snapshot()["swap_now"] is None and "flash-next" not in max_mode.snapshot()["inflight_since"]
          and out.get("swap", {}).get("to") == "flash-next",
          "afterwards nothing is in progress and the lease age is gone", max_mode.snapshot())


def test_each_slot_says_its_conversation_and_hold():
    """slots.occupants -> lane_view: the conversation in each slot, idle for how long, the 60 s hold left."""
    reset()
    _two_cards()
    now = time.time()
    k_main, k_other = "aaaa1111bbbb2222", "cccc3333dddd4444"
    saved = (dict(slot_map._pins), dict(slot_map._last_seen), dict(slot_map._used), dict(slot_map._other))
    try:
        slot_map._pins.clear()
        slot_map._pins[k_main] = 0
        slot_map._last_seen[k_main] = now - 20
        slot_map._other.update({"owner": k_other, "busy": 0, "last_end": now - 90, "model": "bonsai-a4000"})
        slot_map._last_seen[k_other] = now - 90
        occ = slot_map.occupants()
        m = occ["main"][0]
        check(m["conv"] == "aaaa1111" and m["busy"] is False and 19 <= m["idle_s"] <= 22 and m["held"] is True
              and 38 <= m["rest_s"] <= 41,
              "occupants: the main card's slot 0 has its conversation, 20 s idle and ~40 s of the 60 s hold left", m)
        o = occ["other"]
        check(o["slot"] == 0 and o["conv"] == "cccc3333" and o["held"] is False and o["rest_s"] == 0,
              "the other card's owner is listed after its hold ended (its cells are still there): held False, 0 left", o)
        by = {c["key"]: c for c in lane_view.snapshot()["cards"]}
        ms = by["main"]["models"][0]["slots"]["slots"][0]
        bon = next(m for m in by["a4000"]["models"] if m["model"] == "bonsai-a4000")
        a0, a1 = bon["slots"]["slots"][0], bon["slots"]["slots"][1]
        check(ms["conv"] and ms["conv"]["conv"] == "aaaa1111" and ms["conv"]["hold_s"] == slot_map.PRIMARY_HOLD_S,
              "the lane view puts it on the main card's slot, with the hold's length", ms.get("conv"))
        check(a0["conv"] and a0["conv"]["conv"] == "cccc3333" and a1["conv"] is None,
              "and the other card's owner on bonsai-a4000 slot 0 only, never on the jjava slot", (a0.get("conv"), a1.get("conv")))
        slot_map._busy[0] = slot_map._busy.get(0, 0) + 1
        try:
            b = slot_map.occupants()["main"][0]
        finally:
            slot_map._busy[0] -= 1
        check(b["busy"] is True and b["rest_s"] is None and b["held"] is True,
              "a request in flight: busy, held, no countdown", b)
    finally:
        slot_map._pins.clear(); slot_map._pins.update(saved[0])
        slot_map._last_seen.clear(); slot_map._last_seen.update(saved[1])
        slot_map._used.clear(); slot_map._used.update(saved[2])
        slot_map._other.clear(); slot_map._other.update(saved[3])


def main() -> int:
    for fn in (test_slots_follow_the_loaded_model, test_a_model_that_is_starting_is_not_read,
               test_context_follows_the_loaded_model, test_the_line_says_what_holds_what,
               test_cards_models_and_slots, test_holds_in_plain_words, test_swaps_view,
               test_pulse_carries_the_view, test_the_power_reader_follows_the_card,
               test_max_mode_records_the_swap_in_progress, test_each_slot_says_its_conversation_and_hold):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised", traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name + (f"   <- {detail}" if not ok and detail else ""))
    check(not _leaks, "no test reached for a real server", str(_leaks))
    for ok, name, detail in _results[-1:]:
        print(("  pass  " if ok else "  FAIL  ") + name + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
