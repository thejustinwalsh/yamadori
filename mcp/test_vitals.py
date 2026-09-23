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


# ------------------------------------------------------------------- slots

def test_the_fixture_cannot_reach_a_server():
    r = vitals.slots()
    check(r["ok"] is False and "blocked" in r["error"], "an unreachable server is ok: False, not a raise", str(r))
    check(_leaks and _leaks[-1].endswith("/slots"), "and the read went to /slots", str(_leaks[-1:]))
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
               test_snapshot_carries_the_new_fields):
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
