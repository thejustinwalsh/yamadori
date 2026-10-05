#!/usr/bin/env python
"""vitals.pulse() and its parts, asserted. No network, no model server, no GPU.

WHAT THIS IS GATING

The tokonoma polls /dash/api/vitals/pulse every second and animates from it.
That is only safe if every part is cheap, bounded and read-only:

  slots()   parses llama-server /slots into idle / prefill / decode with
            tokens per second, asks with a timeout of at most ~1 s, and turns a
            server that is not answering into {"ok": False}, never a raise
  tools()   reads the NEWEST corpus rows read-only: it must not create a
            missing database, must not change an existing one, and must say
            which tool is running when the newest event is an unanswered call
  queue()   job counts, read-only
  lanes()   admission's in-flight counters, as the proxy holds them
  pulse()   all of the above plus GPUs, and the /dash/api/vitals/pulse route
  snapshot() carries the same new fields

Every database here is a temp file; `urllib.request.urlopen` is replaced so
nothing can reach :10001, and nvidia-smi is replaced so no process is started.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import traceback
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_vitals_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")
os.environ["CONCEPT_SEED_LAST"] = os.path.join(_TMP, "seed_last.json")

import admission  # noqa: E402
import corpus  # noqa: E402
import dash_vitals  # noqa: E402
import jobs  # noqa: E402
import vitals  # noqa: E402

# budget.budgets() asks the model server for n_ctx once; answer it here.
import budget  # noqa: E402
budget._POOL = 147456

_results: list[tuple[bool, str, str]] = []
_leaks: list[str] = []
_calls: list[tuple[str, float | None]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def _blocked(url, timeout=None):                                  # noqa: ARG001
    _leaks.append(url if isinstance(url, str) else url.full_url)
    raise OSError("test_vitals: network is blocked")


urllib.request.urlopen = _blocked
_real_sh = vitals._sh
vitals._sh = lambda cmd, timeout=25: (                            # noqa: E731
    "0, NVIDIA GeForce RTX 5060 Ti, 12000, 16311, 87\n"
    "1, NVIDIA RTX A4000, 7000, 16376, 3\n" if cmd and cmd[0] == "nvidia-smi" else "")


def _slots_json(decoded: int, processed: int = 3000, busy: bool = True, task: int = 7):
    return json.dumps([
        {"id": 0, "n_ctx": 163840, "is_processing": busy, "id_task": task,
         "n_prompt_tokens": processed + decoded, "n_prompt_tokens_processed": processed,
         "next_token": [{"has_next_token": busy, "n_remain": 1000, "n_decoded": decoded}]},
        {"id": 1, "n_ctx": 163840, "is_processing": True, "id_task": 9,
         "n_prompt_tokens": 4000, "n_prompt_tokens_processed": 1200,
         "next_token": [{"has_next_token": True, "n_remain": -1, "n_decoded": 0}]},
        {"id": 2, "n_ctx": 163840, "is_processing": False, "id_task": 3,
         "n_prompt_tokens": 0, "n_prompt_tokens_processed": 0,
         "next_token": [{"has_next_token": False, "n_remain": -1, "n_decoded": 0}]},
    ]).encode()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _serve(body: bytes):
    def fake(url, timeout=None):
        _calls.append((url if isinstance(url, str) else url.full_url, timeout))
        return _Resp(body)
    return fake


def restore() -> None:
    urllib.request.urlopen = _blocked
    vitals._slot_prev.clear()
    vitals._slot_rate.clear()
    vitals._running_cache = (-1e9, None)


# ------------------------------------------------------------------- slots

def test_the_fixture_cannot_reach_a_server():
    r = vitals.slots()
    check(r["ok"] is False and "blocked" in r["error"], "an unreachable server is ok: False, not a raise", str(r))
    check(any(u.endswith("/slots") for u in _leaks), "and the read went to /slots", str(_leaks[-3:]))
    # Why it does not answer, in plain words (2026-10-05: the line read "bonsai is off the card: <the coordinator's
    # note> holds it" -- a PAUSED JOB LANE named as the card's holder). Nothing loaded: say so, and say the lanes'
    # pause is a pause on the worker's queue, not a hold on the card.
    import jobs
    real = jobs.paused
    jobs.paused = lambda lane: ({"by": "coordinator", "why": "operator sessions 2026-10-02: held until the skills window",
                                 "since": 1.0, "until": 4102444800.0} if lane == "gpu" else None)
    _running([])
    try:
        r = vitals.slots()
    finally:
        jobs.paused = real
        _running(None)
    check(r["ok"] is False and r.get("off_card") is True and "no main model is loaded" in r["error"]
          and "blocked" in r.get("cause", ""),
          "nothing loaded: named as that, the underlying failure kept as `cause`", str(r))
    check("holds it" not in r["error"] and "paused by coordinator" in r["error"]
          and "not held by it" in r["error"] and "operator sessions" not in r["error"],
          "a paused job lane is said as a pause on the worker's queue, never as the card's holder", str(r))
    _leaks.clear()


def test_slot_states_and_rates():
    now = [1000.0]
    real_time = vitals.time.time
    vitals.time.time = lambda: now[0]
    try:
        urllib.request.urlopen = _serve(_slots_json(decoded=100))
        a = vitals.slots()
        now[0] += 1.0
        urllib.request.urlopen = _serve(_slots_json(decoded=142, processed=3000))
        b = vitals.slots()
    finally:
        vitals.time.time = real_time
    s = {x["id"]: x for x in b["slots"]}
    check(a["ok"] and b["ok"], "a good /slots answer is ok")
    check(s[0]["state"] == "decode", "a slot with decoded tokens is decoding", str(s[0]))
    check(s[1]["state"] == "prefill", "a busy slot with none decoded is in prefill", str(s[1]))
    check(s[2]["state"] == "idle", "a slot not processing is idle", str(s[2]))
    check(abs(s[0]["tps"] - 42.0) < 0.01, "decode tokens/s comes from the change in n_decoded", str(s[0]["tps"]))
    check(s[0]["ctx"] == 3142, "context held counts prompt and decoded tokens", str(s[0]["ctx"]))
    check(b["decoding"] == 1 and b["prefilling"] == 1, "totals count decoding and prefilling slots", str(b))
    check(abs(b["tps"] - 42.0) < 0.01, "and sum the decode rate", str(b["tps"]))
    check(all(t is not None and t <= 1.5 for _, t in _calls[-2:]),
          "every /slots read carries a timeout of at most 1.5 s", str(_calls[-2:]))


def test_a_new_task_restarts_the_rate():
    now = [2000.0]
    real_time = vitals.time.time
    vitals.time.time = lambda: now[0]
    try:
        urllib.request.urlopen = _serve(_slots_json(decoded=500, task=7))
        vitals.slots()
        now[0] += 1.0
        urllib.request.urlopen = _serve(_slots_json(decoded=10, task=8))
        b = vitals.slots()
    finally:
        vitals.time.time = real_time
    s0 = [x for x in b["slots"] if x["id"] == 0][0]
    check(s0["tps"] == 0.0, "a slot that moved to a new task reports no rate, not a negative one", str(s0))


def test_a_malformed_answer_is_named():
    urllib.request.urlopen = _serve(b'{"error": "nope"}')
    r = vitals.slots()
    check(r["ok"] is False and "shape" in r["error"], "a non-list /slots answer is ok: False with a reason", str(r))


# ------------------------------------------------------------------- tools

def _digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_tools_on_a_missing_corpus_creates_nothing():
    missing = os.path.join(_TMP, "absent.sqlite3")
    r = vitals.tools(db=missing)
    check(r is None, "an unreadable corpus is None", str(r))
    check(not os.path.exists(missing), "and reading it did not create the file")


def test_tools_reads_the_newest_rows_read_only():
    t = corpus.new_turn()
    corpus.log_turn(t, None, [{"role": "user", "content": "where is X"}], ["find_by_meaning"], True)
    corpus.log_tool_call(t, None, "find_by_meaning", {"query": "X"}, 0)
    corpus.log_tool_result(t, None, "find_by_meaning", "== TAPROOT ==\n### a.ts:1\n", 12.5)
    corpus.log_tool_call(t, None, "read_file_range", {"path": "a.ts"}, 1)
    before = _digest(corpus.CORPUS_DB)
    r = vitals.tools()
    check(r is not None, "the corpus is readable")
    if r is None:
        return
    check(r["last"] and r["last"]["name"] == "read_file_range", "the last call is the newest tool_call", str(r["last"]))
    check(r["running"] and r["running"]["name"] == "read_file_range",
          "a call with no result yet is reported as running", str(r["running"]))
    check(r["recent"] and r["recent"][0]["name"] == "find_by_meaning" and r["recent"][0]["ms"] == 12.5,
          "recent results carry their latency", str(r["recent"]))
    check(r["calls_window"] == 2 and r["turns_window"] == 1, "calls and requests in the window are counted", str(r))
    check(r["last_turn"] and r["last_turn"]["open"] is True, "an unanswered request is open", str(r["last_turn"]))
    check(_digest(corpus.CORPUS_DB) == before, "reading changed nothing in the corpus file")

    corpus.log_tool_result(t, None, "read_file_range", "1: x", 3.0)
    corpus.log_answer(t, None, "a.ts", 2, 900)
    r = vitals.tools()
    check(r["running"] is None, "once its result lands, nothing is running", str(r["running"]))
    check(r["last_turn"]["open"] is False, "an answered request is closed", str(r["last_turn"]))


# ------------------------------------------------------------------- queue

def test_queue_counts_read_only():
    jobs.add("probe", {"x": 1}, lane="gpu")
    jobs.add("probe", {"x": 2}, lane="cpu")
    before = _digest(jobs.DB)
    q = vitals.queue()
    check(q is not None and q["states"]["queued"] == 2, "queued jobs are counted", str(q))
    check(q is not None and q["oldest_queued_age"] is not None, "and the oldest one's age is given", str(q))
    check(_digest(jobs.DB) == before, "reading changed nothing in the jobs file")
    check(vitals.queue(db=os.path.join(_TMP, "nojobs.sqlite3")) is None
          and not os.path.exists(os.path.join(_TMP, "nojobs.sqlite3")),
          "a missing jobs db is None and is not created")


# ------------------------------------------------------------------- lanes

def test_lanes_follow_admission():
    admission._inflight["main"] += 1
    admission._helper_stats["inflight"] += 1
    try:
        lanes = vitals.lanes()
    finally:
        admission._inflight["main"] -= 1
        admission._helper_stats["inflight"] -= 1
    check(lanes and lanes["main"] == 1 and lanes["helper"] == 1,
          "held main and helper lanes are reported", str(lanes))
    check(lanes and lanes["main_lanes"] == admission.MAIN_LANES, "with the lane counts", str(lanes))


# ------------------------------------------------------------------- pulse

def test_pulse_and_its_route():
    urllib.request.urlopen = _serve(_slots_json(decoded=5))
    p = vitals.pulse()
    for k in ("at", "gpus", "slots", "lanes", "tools", "queue", "seed", "strata", "context"):
        check(k in p, f"pulse carries {k}")
    check(len(p["gpus"]) == 2 and p["gpus"][0]["util"] == 87, "pulse carries GPU util", str(p["gpus"][:1]))
    code, ctype, body = dash_vitals.handle_get("/dash/api/vitals/pulse")
    j = json.loads(body)
    check(code == 200 and ctype == "application/json" and "slots" in j,
          "/dash/api/vitals/pulse answers with the pulse", f"{code} {ctype}")
    check(dash_vitals.handle_get("/dash/api/vitals/other") is None, "other sub-paths are not claimed")


def test_pulse_caches_gpus():
    calls = []

    def sh(cmd, timeout=25):
        calls.append((cmd[0], timeout))
        return "0, GPU, 1, 2, 3\n"
    vitals._sh = sh
    vitals._gpu_cache = (0.0, [])
    urllib.request.urlopen = _serve(_slots_json(decoded=5))
    try:
        vitals.pulse()
        vitals.pulse()
    finally:
        vitals._sh = lambda cmd, timeout=25: ""
    check(len(calls) == 1, "two pulses inside the TTL run nvidia-smi once", str(calls))
    check(calls and calls[0][1] <= 3, "and with a short timeout", str(calls))


def test_snapshot_carries_the_new_fields():
    for name in ("processes", "listeners", "endpoints", "context_pool"):
        setattr(vitals, name, (lambda: []) if name != "context_pool" else (lambda: {"pool": 1}))
    urllib.request.urlopen = _serve(_slots_json(decoded=5))
    s = vitals.snapshot()
    for k in ("slots", "lanes", "tools", "queue"):
        check(k in s, f"snapshot carries {k}")


# ------------------------------------------------ the dashboard fixup (2026-09-29)

# test_snapshot_carries_the_new_fields replaces these on the module; keep the
# real ones (this runs at import, before any test).
_REAL = {n: getattr(vitals, n) for n in ("endpoints", "context_pool")}

def test_process_start_times_parse():
    # Windows PowerShell 5.1's ConvertTo-Json writes "/Date(<ms>)/"; the old
    # [:19] cut put "/Date(1790680397077" on the page.
    got = vitals._started("/Date(1790680397077)/")
    check(len(got) == 19 and got[4] == "-" and got[10] == "T",
          "a /Date(ms)/ start time becomes local ISO seconds", got)
    check(vitals._started("2026-09-29T07:01:02.123+00:00") == "2026-09-29T07:01:02",
          "an ISO start time is cut to seconds, as before")
    check(vitals._started(None) == "" and vitals._started("/Date(nope)/") == "/Date(nope)/",
          "nothing and a malformed value never raise")


class _Ok(_Resp):
    status = 200


def test_watched_services():
    roles = set(vitals.PORTS.values())
    check({"proxy", "tools-api", "searxng", "llama-swap", "bonsai"} <= roles,
          "the listeners cover every supervised service with a port", str(sorted(roles)))
    check("laya" not in roles, "Laya (retired) is not watched")
    names = [n for _, n in vitals.PROBES]
    check(names == ["llama-swap", "tools-api", "searxng"], "the probed endpoints", str(names))
    check(all(u.endswith(("/v1/models", "/health", "/healthz")) for u, _ in vitals.PROBES),
          "each probe is a liveness route that loads nothing", str(vitals.PROBES))
    seen = []

    def fake(url, timeout=None):
        seen.append((url if isinstance(url, str) else url.full_url, timeout))
        if "8888" in (url if isinstance(url, str) else url.full_url):
            raise OSError("searxng down")
        return _Ok(b"{}")
    urllib.request.urlopen = fake
    rows = _REAL["endpoints"]()
    by = {r["name"]: r for r in rows}
    check(by["llama-swap"]["ok"] and by["tools-api"]["ok"], "answering services are ok", str(rows))
    check(by["searxng"]["ok"] is False and by["searxng"]["code"] == 0,
          "a service that does not answer is down, the others unaffected", str(by["searxng"]))
    check(all(t is not None and t <= vitals.PROBE_TIMEOUT for _, t in seen),
          "every probe carries a timeout", str(seen))


def _running(rows):
    import time as _t
    vitals._running_cache = (_t.time(), rows)


def test_serving_names_the_model_on_the_card():
    import max_mode
    _running([{"model": "bonsai", "state": "ready", "proxy": "http://127.0.0.1:10001",
               "cmd": "llama-server --port 10001 -m C:\\models\\Ternary-Bonsai-2-27B.gguf --mmproj x-mmproj.gguf"},
              {"model": "embed", "state": "ready", "proxy": "http://127.0.0.1:10006",
               "cmd": "llama-server -m /m/Qwen3-Embedding-0.6B-Q8_0.gguf"}])
    s = vitals.serving()
    check(s["on_card"] == "bonsai" and s["main"] == "bonsai", "the main model is named on the card", str(s))
    check(s["enabled"] is max_mode.ENABLED, "max mode's switch is max_mode's own", str(s["enabled"]))
    row = [x for x in s["loaded"] if x["model"] == "bonsai"][0]
    check(row["port"] == 10001 and row["gguf"] == "Ternary-Bonsai-2-27B.gguf",
          "a loaded row carries its port and its -m gguf, not the mmproj", str(row))
    _running(None)
    s = vitals.serving()
    check(s["loaded"] is None and s["on_card"] is None,
          "llama-swap unreadable: loaded is None, nothing is claimed", str(s))
    real = (max_mode.ENABLED, max_mode.MAX)
    max_mode.ENABLED, max_mode.MAX = True, "flash-next"
    real_snap, real_active = max_mode.snapshot, max_mode.max_active
    max_mode.max_active = lambda: True
    max_mode.snapshot = lambda: {"enabled": True, "main": "bonsai", "max": "flash-next",
                                 "inflight": {"flash-next": 1}, "switching_to": None,
                                 "last_max_end": 0.0, "idle_s": None}
    try:
        _running([{"model": "flash-next", "state": "ready", "proxy": "http://127.0.0.1:10009",
                   "cmd": "llama-server -m /m/Qwen3.8-Flash-Next-IQ2_XS-00001-of-00002.gguf"}])
        s = vitals.serving()
    finally:
        max_mode.ENABLED, max_mode.MAX = real
        max_mode.snapshot, max_mode.max_active = real_snap, real_active
    check(s["on_card"] == "flash-next" and s["max"] == "flash-next" and s["max_active"] is True,
          "in max mode the max model is on the card and max mode is active", str(s))


def test_max_mode_slots_and_off_card():
    import max_mode
    real = (max_mode.ENABLED, max_mode.MAX)
    max_mode.ENABLED, max_mode.MAX = True, "flash-next"
    try:
        _running([{"model": "flash-next", "state": "ready", "proxy": "http://127.0.0.1:10009"}])
        url, model = vitals._slots_target()
        check(url == "http://127.0.0.1:10009/slots" and model == "flash-next",
              "while the max model holds the card /slots is read from its port", url)
        check(vitals.off_card_why() is None,
              "a ready max model is the card's model: nothing is 'off the card'", str(vitals.off_card_why()))
        _running([{"model": "bonsai", "state": "ready", "proxy": "http://127.0.0.1:10001"}])
        check(vitals._slots_target() == (vitals.SLOTS_URL, vitals.MAIN_MODEL),
              "the main model on the card: its own /slots")
    finally:
        max_mode.ENABLED, max_mode.MAX = real
    _running([{"model": "flash-next", "state": "ready", "proxy": "http://127.0.0.1:10009"}])
    check(vitals._slots_target() == (vitals.SLOTS_URL, vitals.MAIN_MODEL),
          "max mode off: never another model's port")


def test_slot_roles():
    import slots as slot_map
    real = slot_map.snapshot
    slot_map.snapshot = lambda: {"pins": {"aaaaaaaa": 0, "bbbbbbbb": 1},
                                 "ranks": {"primary": "bbbbbbbb"}}
    urllib.request.urlopen = _serve(_slots_json(decoded=5))
    try:
        r = vitals.slots()
    finally:
        slot_map.snapshot = real
    by = {x["id"]: x for x in r["slots"]}
    check(by[2]["role"] == "child" and by[0]["role"] == by[1]["role"] == "conversation",
          "three slots: the last is the child, the rest conversations", str(r["slots"]))
    check(by[0]["pinned"] and by[1]["pinned"] and not by[2]["pinned"],
          "pins mark the conversation slots in use", str(r["slots"]))
    check(by[1]["primary"] and not by[0]["primary"], "the primary conversation's slot is marked")


def test_context_carries_the_vram_line():
    real = budget._LINE
    budget._LINE = 141824
    try:
        c = vitals._with_line(budget.budgets())
    finally:
        budget._LINE = real
    check(c.get("vram_line") == 141824, "context carries the served VRAM line", str(c))
    check(vitals._with_line({"error": "x"}) == {"error": "x"}, "an error context is passed through")


def main() -> int:
    for fn in (test_the_fixture_cannot_reach_a_server,
               test_slot_states_and_rates,
               test_a_new_task_restarts_the_rate,
               test_a_malformed_answer_is_named,
               test_tools_on_a_missing_corpus_creates_nothing,
               test_tools_reads_the_newest_rows_read_only,
               test_queue_counts_read_only,
               test_lanes_follow_admission,
               test_pulse_and_its_route,
               test_pulse_caches_gpus,
               test_snapshot_carries_the_new_fields,
               test_process_start_times_parse,
               test_watched_services,
               test_serving_names_the_model_on_the_card,
               test_max_mode_slots_and_off_card,
               test_slot_roles,
               test_context_carries_the_vram_line):
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
