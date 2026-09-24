#!/usr/bin/env python
"""The watts-vs-tokens ring in mcp/power.py, asserted. No GPU, no network,
no model server: /running and /slots are fakes, the ring is synthetic.

WHAT THIS IS GATING

  SlotCounter   passive: a model llama-swap does not list as ready is never
                asked for /slots; the first read only primes; deltas within
                a task, a new task counted whole, idle slots dropped; a
                failed /slots re-asks /running; localhost is rewritten
  the ring      bounded at RING_SAMPLES, only when a token reader is given
  series_of     tok/s = delta / dt; the idle baseline per card; J/token and
                marginal J/token on the main card only; per-phase J/token
                from single-phase samples; Pearson r with its n, withheld
                below MIN_R_SAMPLES; a not-loaded second has no rate
  the route     /dash/api/power/series
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_power_series_")
os.environ["YAMADORI_POWER_LEDGER"] = os.path.join(_TMP, "ledger.json")

import power  # noqa: E402

_results: list[tuple[bool, str, str]] = []
MAIN = power.MAIN_GPU_UUID
A4000 = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def near(a, b, tol=1e-6) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


class Fake:
    """llama-swap /running and the model's /slots, scripted."""

    def __init__(self):
        self.running = {"running": [{"model": "bonsai", "state": "ready",
                                     "proxy": "http://localhost:10001"}]}
        self.slots: list | Exception = []
        self.calls: list[str] = []

    def get(self, url, timeout):
        self.calls.append(url)
        if url.endswith("/running"):
            if isinstance(self.running, Exception):
                raise self.running
            return self.running
        if isinstance(self.slots, Exception):
            raise self.slots
        return self.slots


def slot(sid, task, decoded, processed, busy=True):
    return {"id": sid, "id_task": task, "is_processing": busy,
            "n_prompt_tokens_processed": processed,
            "next_token": [{"n_decoded": decoded}]}


def test_slot_counter():
    f = Fake()
    clock = iter(range(100, 200)).__next__
    c = power.SlotCounter(get=f.get, clock=clock)
    f.running = {"running": [{"model": "embeddings", "state": "ready", "proxy": "http://x"}]}
    r = c.read()
    check(r["state"] == "not_loaded" and not any(u.endswith("/slots") for u in f.calls),
          "bonsai not listed: not_loaded, and /slots is never asked", str(r))
    f.running = {"running": [{"model": "bonsai", "state": "starting", "proxy": "http://localhost:10001"}]}
    check(c.read()["state"] == "starting", "a model still starting is that state, not a rate")
    f.running = OSError("refused")
    check(c.read()["state"] == "swap_down", "llama-swap not answering is swap_down")
    f.running = {"running": [{"model": "bonsai", "state": "ready", "proxy": "http://localhost:10001"}]}
    f.slots = [slot(0, 7, 500, 9000)]
    f.calls.clear()
    r = c.read()
    check(r["state"] == "ok" and r["decoded"] is None and r["busy"] == 1,
          "the first read only primes (a running task's past is not one second's)", str(r))
    check(any(u == "http://127.0.0.1:10001/slots" for u in f.calls),
          "localhost from /running is asked as 127.0.0.1", str(f.calls))
    f.slots = [slot(0, 7, 530, 9000), slot(1, 9, 3, 1200)]
    r = c.read()
    check(r["decoded"] == 33 and r["prompt"] == 1200 and r["busy"] == 2 and r["dt"] == 1,
          "same task: the difference; a new task: all of it", str(r))
    f.slots = [slot(0, 7, 530, 9000, busy=False), slot(1, 9, 20, 1200)]
    r = c.read()
    check(r["decoded"] == 17 and r["prompt"] == 0 and r["busy"] == 1,
          "an idle slot contributes nothing", str(r))
    f.slots = [slot(0, 11, 4, 50), slot(1, 9, 20, 1200)]
    r = c.read()
    check(r["decoded"] == 4 and r["prompt"] == 50,
          "a slot that was idle and is busy again counts its new task whole", str(r))
    f.slots = OSError("timed out")
    f.calls.clear()
    r = c.read()
    check(r["state"] == "no_answer", "/slots failing is no_answer", str(r))
    f.slots = []
    c.read()
    check(any(u.endswith("/running") for u in f.calls),
          "and the next read asks /running again", str(f.calls))


def rows(main_w, a4000_w):
    return [{"index": 0, "name": "NVIDIA GeForce RTX 5060 Ti", "uuid": MAIN, "watts": main_w},
            {"index": 1, "name": "NVIDIA RTX A4000", "uuid": A4000, "watts": a4000_w}]


def test_ring():
    s = power.Sampler(read=lambda: rows(20.0, 6.0), path=os.path.join(_TMP, "a.json"))
    s.sample_once()
    check(s.ring() == [], "no token reader: nothing enters the ring")
    n = iter(range(1000)).__next__
    s2 = power.Sampler(read=lambda: rows(20.0, 6.0), path=os.path.join(_TMP, "b.json"),
                       clock=lambda: float(n()),
                       read_tokens=lambda: {"state": "ok", "busy": 0, "decoded": 0, "prompt": 0, "dt": 1.0})
    for _ in range(power.RING_SAMPLES + 50):
        s2.sample_once()
    ring = s2.ring()
    check(len(ring) == power.RING_SAMPLES, "the ring is bounded", str(len(ring)))
    check(ring[-1]["gpus"][0]["watts"] == 20.0 and ring[-1]["tok"]["state"] == "ok",
          "an entry holds every card's watts and the token reading")
    s3 = power.Sampler(read=lambda: rows(1.0, 1.0), path=os.path.join(_TMP, "c.json"),
                       read_tokens=lambda: 1 / 0)
    s3.sample_once()
    check(s3.ring()[-1]["tok"]["state"] == "error", "a reader that raises is an error entry")


def synthetic(now=1000.0):
    """60 idle s (20 W / 6 W), 30 decode-only s (130 W, 30 tok/s), 10
    prefill-only s (150 W, 1000 tok/s), 5 not-loaded s."""
    ring, t = [], now - 105
    for _ in range(60):
        t += 1
        ring.append({"t": t, "gpus": rows(20.0, 6.0),
                     "tok": {"state": "ok", "busy": 0, "decoded": 0, "prompt": 0, "dt": 1.0}})
    for _ in range(30):
        t += 1
        ring.append({"t": t, "gpus": rows(130.0, 6.0),
                     "tok": {"state": "ok", "busy": 1, "decoded": 30, "prompt": 0, "dt": 1.0}})
    for _ in range(10):
        t += 1
        ring.append({"t": t, "gpus": rows(150.0, 6.0),
                     "tok": {"state": "ok", "busy": 1, "decoded": 0, "prompt": 1000, "dt": 1.0}})
    for _ in range(5):
        t += 1
        ring.append({"t": t, "gpus": rows(8.0, 6.0), "tok": {"state": "not_loaded", "dt": 1.0}})
    return ring, now


def test_series_math():
    ring, now = synthetic()
    d = power.series_of(ring, now)
    check(d["gpus"][0]["uuid"] == MAIN and d["gpus"][0]["main"] and d["main_index"] == 0,
          "the main card is first and marked")
    check(len(d["t"]) == 105 and d["t"][-1] == 0.0 and len(d["watts"][1]) == 105,
          "aligned columns, time relative to now", str(len(d["t"])))
    check(d["decode_tps"][70] == 30.0 and d["prompt_tps"][95] == 1000.0,
          "tok/s = delta / dt, per phase")
    check(d["decode_tps"][-1] is None and d["busy"][-1] is None,
          "a not-loaded second has no rate (not zero)")
    idle = d["idle"]
    check(idle["watts"] == [20.0, 6.0] and idle["n"] == 60,
          "idle baseline per card from idle samples only", str(idle))
    st = d["stats"]
    # generating: 30 x 130 J for 900 tokens + 10 x 150 J for 10,000 tokens
    check(near(st["j_per_token"], (30 * 130 + 10 * 150) / 10900) and st["tokens"] == 10900,
          "J/token = sum(main W x dt) / tokens over generating samples", str(st["j_per_token"]))
    check(near(st["marginal_j_per_token"], (30 * 110 + 10 * 130) / 10900),
          "marginal J/token subtracts the idle main-card draw", str(st["marginal_j_per_token"]))
    check(near(st["j_per_gen_token"], 130 / 30) and st["gen_only_samples"] == 30,
          "J per generated token from decode-only samples", str(st["j_per_gen_token"]))
    check(near(st["j_per_prompt_token"], 0.15) and st["prompt_only_samples"] == 10,
          "J per prompt token from prefill-only samples", str(st["j_per_prompt_token"]))
    check(st["r_n"] == 100 and st["r_decode"] is not None and st["r_decode"] > 0.5,
          "r(main W, decode tok/s) over the 100 answered samples", f"{st['r_decode']} n={st['r_n']}")
    check(st["r_prompt"] is not None and st["r_prompt"] > 0,
          "r against prompt tok/s too", str(st["r_prompt"]))
    check(d["model"]["state"] == "not_loaded", "the newest state is reported")
    few = power.series_of(ring[:20], ring[19]["t"])
    check(few["stats"]["r_decode"] is None and few["stats"]["r_n"] == 20,
          "below MIN_R_SAMPLES no coefficient is given", str(few["stats"]["r_decode"]))
    check(few["stats"]["j_per_token"] is None,
          "no generating sample: no J/token (not zero)")
    busy_only = ring[60:90]
    b = power.series_of(busy_only, busy_only[-1]["t"], idle_prev={"watts": [21.0, 6.0], "n": 40, "at": 1})
    check(b["idle"]["watts"] == [21.0, 6.0] and b["idle"]["from_window"] is False,
          "no idle second in the window: the last baseline, marked as earlier")
    check(near(b["stats"]["marginal_j_per_token"], 109 / 30), "and it is used for the marginal",
          str(b["stats"]["marginal_j_per_token"]))
    cut = power.series_of(ring, now, window=10)
    check(len(cut["t"]) == 11, "the window cuts older samples", str(len(cut["t"])))


def test_series_and_route():
    check("error" in power.series(None) or power.sampler() is not None,
          "no sampler in this process: an error, not an empty chart")
    s = power.Sampler(read=lambda: rows(20.0, 6.0), path=os.path.join(_TMP, "d.json"))
    check("error" in power.series(s), "a sampler with no token reader says so")
    ring, now = synthetic()
    s2 = power.Sampler(read=lambda: rows(20.0, 6.0), path=os.path.join(_TMP, "e.json"),
                       read_tokens=lambda: {})
    for e in ring:
        s2.ring_add(e["t"], e["gpus"], e["tok"])
    out = power.series(s2, now=now)
    check(out["stats"]["samples"] == 105 and s2._idle and s2._idle["n"] == 60,
          "series() reads the ring and keeps the idle baseline for later", str(s2._idle))
    json.dumps(out)
    import dash_vitals
    code, ctype, body = dash_vitals.handle_get("/dash/api/power/series")
    check(code == 200 and ctype == "application/json" and isinstance(json.loads(body), dict),
          "/dash/api/power/series answers", str(code))


def main() -> int:
    for fn in (test_slot_counter, test_ring, test_series_math, test_series_and_route):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
