#!/usr/bin/env python
"""mcp/power.py, asserted. No GPU, no network, no model server.

WHAT THIS IS GATING

  rate_at()          DTE D1.11: summer (June-September) and winter prices,
                     peak Monday-Friday 15:00-19:00 local, 15:00 inclusive,
                     19:00 exclusive, weekends off-peak all day, and the
                     YAMADORI_POWER_RATE_CENTS flat override
  Sampler.ingest()   trapezoid integration into the daily ledger, split by
                     period and by local date; a gap or a card that reports no
                     draw is counted as a gap, never integrated
  the ledger         persists and reloads; a corrupt file is kept aside
  request_energy()   x_yamadori.energy: window mean, rolling fallback, no
                     sampler; and _x_yamadori really carries it
  live()             the /dash/api/power payload shape, and the route
  vitals.gpus()      parses power.draw / uuid, and still parses five columns
  dash_results       LiveBench electricity per question from seconds x watts

nvidia-smi is never started: every reader here is a fake.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_power_")
os.environ["YAMADORI_POWER_LEDGER"] = os.path.join(_TMP, "ledger.json")
os.environ.pop("YAMADORI_POWER_RATE_CENTS", None)
os.environ.pop("YAMADORI_POWER_EXTRA_WATTS", None)

import power  # noqa: E402
import vitals  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def near(a, b, tol=1e-6) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def ep(*a) -> float:
    """Epoch of a naive LOCAL datetime (the sampler's clock is local)."""
    return time.mktime(dt.datetime(*a).timetuple())


def rows(*watts):
    return [{"index": i, "name": f"GPU{i}", "uuid": f"GPU-{i}", "watts": w,
             "watts_limit": 180.0} for i, w in enumerate(watts)]


def new_sampler(name: str) -> power.Sampler:
    return power.Sampler(read=lambda: rows(100.0),
                         path=os.path.join(_TMP, f"{name}.json"))


# Thursday 2026-09-24 is a weekday in summer; 2026-09-26 is a Saturday.
def test_rate_periods():
    r = power.rate_at
    check(near(power._cents(power.RATE_TABLE["summer"]["peak"]), 24.133),
          "summer peak is 5.360+9.047+9.726 = 24.133 c")
    check(near(power._cents(power.RATE_TABLE["summer"]["off_peak"]), 18.435),
          "summer off-peak is 3.240+5.469+9.726 = 18.435 c")
    check(near(power._cents(power.RATE_TABLE["winter"]["peak"]), 20.045),
          "winter peak is 3.839+6.480+9.726 = 20.045 c")
    check(near(power._cents(power.RATE_TABLE["winter"]["off_peak"]), 18.435),
          "winter off-peak is 18.435 c")
    a = r(dt.datetime(2026, 9, 24, 14, 59, 59))
    check(a["period"] == "off_peak" and near(a["cents_per_kwh"], 18.435),
          "weekday 14:59:59 is off-peak", str(a))
    a = r(dt.datetime(2026, 9, 24, 15, 0, 0))
    check(a["period"] == "peak" and near(a["cents_per_kwh"], 24.133),
          "weekday 15:00:00 is peak (the boundary is inclusive)", str(a))
    a = r(dt.datetime(2026, 9, 24, 18, 59, 59))
    check(a["period"] == "peak", "weekday 18:59:59 is still peak", str(a))
    a = r(dt.datetime(2026, 9, 24, 19, 0, 0))
    check(a["period"] == "off_peak", "weekday 19:00:00 is off-peak (exclusive end)", str(a))
    for day, name in ((26, "Saturday"), (27, "Sunday")):
        a = r(dt.datetime(2026, 9, day, 16, 0))
        check(a["period"] == "off_peak" and near(a["cents_per_kwh"], 18.435),
              f"{name} 16:00 is off-peak", str(a))
    a = r(dt.datetime(2026, 9, 28, 16, 0))
    check(a["period"] == "peak", "Monday 16:00 is peak", str(a))
    a = r(dt.datetime(2026, 9, 25, 16, 0))
    check(a["period"] == "peak", "Friday 16:00 is peak", str(a))
    # Season boundaries by month.
    a = r(dt.datetime(2026, 9, 30, 16, 0))
    check(a["season"] == "summer" and near(a["cents_per_kwh"], 24.133),
          "September 30 is summer", str(a))
    a = r(dt.datetime(2026, 10, 1, 16, 0))
    check(a["season"] == "winter" and near(a["cents_per_kwh"], 20.045),
          "October 1 is winter: peak 20.045 c", str(a))
    a = r(dt.datetime(2027, 5, 31, 16, 0))
    check(a["season"] == "winter", "May 31 is winter", str(a))
    a = r(dt.datetime(2027, 6, 1, 16, 0))
    check(a["season"] == "summer" and near(a["cents_per_kwh"], 24.133),
          "June 1 is summer", str(a))
    a = r(dt.datetime(2026, 12, 15, 10, 0))
    check(a["season"] == "winter" and a["period"] == "off_peak"
          and near(a["cents_per_kwh"], 18.435), "December morning is winter off-peak")
    check(a["components"] == power.RATE_TABLE["winter"]["off_peak"]
          and "D1.11" in a["source"], "the price carries its parts and its source")
    # When the period next changes.
    a = r(dt.datetime(2026, 9, 24, 10, 0))
    check(a["until_local"] == "Thu 15:00", "weekday morning: off-peak until 15:00 today", str(a))
    a = r(dt.datetime(2026, 9, 24, 16, 30))
    check(a["until_local"] == "Thu 19:00", "during peak: until 19:00", str(a))
    a = r(dt.datetime(2026, 9, 25, 19, 0))
    check(a["until_local"] == "Mon 15:00", "Friday 19:00: off-peak until Monday 15:00", str(a))
    check(near(a["until"], ep(2026, 9, 28, 15, 0)), "and `until` is that epoch")
    check(power.rate_range() == (18.435, 24.133), "rate range spans the table",
          str(power.rate_range()))


def test_flat_and_extra_overrides():
    os.environ["YAMADORI_POWER_RATE_CENTS"] = "21.5"
    try:
        a = power.rate_at(dt.datetime(2026, 9, 24, 16, 0))
        check(a["period"] == "flat" and a["cents_per_kwh"] == 21.5 and a["flat"],
              "YAMADORI_POWER_RATE_CENTS replaces the table", str(a))
        check(power.rate_range() == (21.5, 21.5), "and the range collapses to it")
    finally:
        os.environ.pop("YAMADORI_POWER_RATE_CENTS")
    os.environ["YAMADORI_POWER_RATE_CENTS"] = "not a number"
    try:
        check(power.rate_at(dt.datetime(2026, 9, 24, 16, 0))["period"] == "peak",
              "an unparseable flat rate is ignored, not a crash")
    finally:
        os.environ.pop("YAMADORI_POWER_RATE_CENTS")
    check(power.extra_watts() == 0.0, "extra watts default to 0")
    os.environ["YAMADORI_POWER_EXTRA_WATTS"] = "85"
    try:
        check(power.extra_watts() == 85.0, "YAMADORI_POWER_EXTRA_WATTS is read")
        check(power.basis()["extra_watts"] == 85.0, "and the basis labels it")
    finally:
        os.environ.pop("YAMADORI_POWER_EXTRA_WATTS")


def test_integration():
    s = new_sampler("integ")
    t0 = ep(2026, 9, 24, 10, 0, 0)            # weekday, off-peak
    for i in range(3601):
        s.ingest(t0 + i, rows(90.0, 10.0))
    d = s.ledger["days"]["2026-09-24"]
    check(near(d["gpu_wh"], 100.0, 1e-6), "100 W for one hour is 100 Wh", str(d["gpu_wh"]))
    check(near(d["wh"]["off_peak"], 100.0, 1e-6) and "peak" not in d["wh"],
          "all of it off-peak at 10:00", str(d["wh"]))
    check(near(d["cents"]["off_peak"], 0.1 * 18.435, 1e-6),
          "0.1 kWh x 18.435 c = 1.8435 c", str(d["cents"]))
    check(near(d["seconds"], 3600.0), "3600 measured seconds")

    s = new_sampler("trap")
    s.ingest(t0, rows(0.0))
    s.ingest(t0 + 2, rows(100.0))
    check(near(s.ledger["days"]["2026-09-24"]["gpu_wh"], 100.0 / 3600.0),
          "a 0 -> 100 W ramp over 2 s is the trapezoid, 100 J")

    s = new_sampler("gap")
    s.ingest(t0, rows(100.0))
    s.ingest(t0 + 1, rows(100.0))
    s.ingest(t0 + 1 + power.MAX_GAP_SECONDS + 5, rows(100.0))
    d = s.ledger["days"]["2026-09-24"]
    check(near(d["gpu_wh"], 100.0 / 3600.0), "a gap over MAX_GAP_SECONDS is not integrated",
          str(d))
    check(near(d["gap_seconds"], power.MAX_GAP_SECONDS + 5), "it is counted as gap_seconds")

    s = new_sampler("na")
    s.ingest(t0, rows(100.0, None))
    s.ingest(t0 + 1, rows(100.0, 5.0))
    d = s.ledger["days"]["2026-09-24"]
    check(d["gpu_wh"] == 0 and near(d["gap_seconds"], 1.0),
          "a card with no power reading makes the interval a gap, not a partial sum", str(d))
    check(power.total_watts(rows(100.0, None)) is None and power.total_watts([]) is None,
          "total_watts is None for a missing card or no cards")

    s = new_sampler("split")
    start = ep(2026, 9, 24, 14, 59, 0)
    for i in range(121):
        s.ingest(start + i, rows(120.0))
    d = s.ledger["days"]["2026-09-24"]
    check(near(d["wh"]["off_peak"], 2.0) and near(d["wh"]["peak"], 2.0),
          "one minute either side of 15:00 splits 2 Wh off-peak / 2 Wh peak", str(d["wh"]))
    check(near(d["cents"]["peak"], 0.002 * 24.133) and near(d["cents"]["off_peak"], 0.002 * 18.435),
          "each half priced at its own rate", str(d["cents"]))

    s = new_sampler("midnight")
    start = ep(2026, 9, 24, 23, 59, 0)
    for i in range(121):
        s.ingest(start + i, rows(60.0))
    days = s.ledger["days"]
    check(near(days["2026-09-24"]["gpu_wh"], 1.0) and near(days["2026-09-25"]["gpu_wh"], 1.0),
          "a run across midnight is split by local date", str(days))

    os.environ["YAMADORI_POWER_EXTRA_WATTS"] = "50"
    try:
        s = new_sampler("extra")
        for i in range(3601):
            s.ingest(t0 + i, rows(100.0))
        d = s.ledger["days"]["2026-09-24"]
        check(near(d["gpu_wh"], 100.0) and near(d["extra_wh"], 50.0)
              and near(d["wh"]["off_peak"], 150.0),
              "extra watts are integrated and recorded separately", str(d))
    finally:
        os.environ.pop("YAMADORI_POWER_EXTRA_WATTS")

    s = new_sampler("backwards")
    s.ingest(t0 + 10, rows(100.0))
    s.ingest(t0 + 5, rows(100.0))
    check(len(s._t) == 1, "a sample older than the last is dropped, not integrated negative")


def test_ledger_persists():
    path = os.path.join(_TMP, "persist.json")
    s = power.Sampler(read=lambda: rows(100.0), path=path)
    t0 = ep(2026, 9, 24, 16, 0, 0)
    for i in range(61):
        s.ingest(t0 + i, rows(60.0))
    s.save()
    check(os.path.exists(path) and not os.path.exists(path + ".tmp"),
          "the ledger is written atomically")
    data = json.load(open(path, encoding="utf-8"))
    check("D1.11" in data.get("source", ""), "the file names its rate source")
    s2 = power.Sampler(read=lambda: rows(100.0), path=path)
    check(near(s2.ledger["days"]["2026-09-24"]["wh"]["peak"], 1.0),
          "a new sampler reloads it (restart keeps the day)", str(s2.ledger))
    s2.ingest(t0 + 1000, rows(60.0))
    check(near(s2.ledger["days"]["2026-09-24"]["wh"]["peak"], 1.0),
          "the first sample after a restart integrates nothing across the downtime")
    bad = os.path.join(_TMP, "bad.json")
    open(bad, "w").write("{not json")
    s3 = power.Sampler(read=lambda: rows(1.0), path=bad)
    kept = [f for f in os.listdir(_TMP) if f.startswith("bad.json.bad-")]
    check(s3.ledger == {"version": 1, "days": {}} and kept,
          "a corrupt ledger is kept aside, and a fresh one started", str(kept))


def test_request_energy():
    s = new_sampler("req")
    now = time.time()
    for i in range(120):
        s.ingest(now - 120 + i, rows(130.0 if i >= 60 else 10.0))
    e = power.request_energy(now - 60, now, s=s)
    for k in ("seconds", "gpu_watts_avg", "wh", "cents", "rate_period"):
        check(k in e, f"x_yamadori.energy has `{k}`", str(e))
    check(e["basis"] == "window" and e["samples"] == 60, "a long request uses its own window", str(e))
    check(near(e["gpu_watts_avg"], 130.0, 0.05), "its mean is the window's draw, not the day's", str(e))
    check(near(e["wh"], 130.0 * 60 / 3600, 1e-3), "Wh = watts x seconds / 3600", str(e))
    check(near(e["cents"], e["wh"] / 1000 * e["cents_per_kwh"], 1e-4), "cents = kWh x the rate")
    check(e["rate_period"] == power.rate_at(now)["period"], "priced at the period now")
    e = power.request_energy(now - 0.2, now - 0.1, s=s)
    check(e["basis"] == "rolling" and e["gpu_watts_avg"] is not None,
          "a request shorter than a sample gets the rolling mean, and says so", str(e))
    e = power.request_energy(now - 5, now, s=None)
    check(e["basis"] == "no_sampler" and e["wh"] is None and e["seconds"] == 5.0,
          "no sampler: seconds still recorded, energy None, not zero", str(e))
    empty = new_sampler("empty")
    e = power.request_energy(now - 5, now, s=empty)
    check(e["basis"] == "no_samples" and e["cents"] is None, "no samples at all: None", str(e))
    check(power.request_energy(None) is None, "no start time: None")


def test_x_yamadori_carries_energy():
    import proxy
    payload = {"_tier": {"name": "medium"}, "_t_start": time.time() - 3}
    x = proxy._x_yamadori(payload, hops=1, fan=None, think=None)
    e = x.get("energy")
    check(isinstance(e, dict) and e.get("seconds") is not None and e["seconds"] >= 3,
          "_x_yamadori carries energy with the request's wall seconds", str(e))
    check(set(("seconds", "gpu_watts_avg", "wh", "cents", "rate_period")) <= set(e or {}),
          "with every field the brief names", str(e))
    x = proxy._x_yamadori({"_tier": {"name": "medium"}}, hops=1, fan=None, think=None)
    check("energy" in x and x["energy"] is None, "a payload with no start time: energy None")


def test_live_payload():
    s = new_sampler("live")
    now = time.time()
    for i in range(30):
        s.ingest(now - 30 + i, rows(132.3, 6.2))
    p = power.live(s, now=now)
    for k in ("running", "gpus", "gpu_watts", "extra_watts", "total_watts",
              "rate", "dollars_per_hour", "today", "days", "basis"):
        check(k in p, f"live() has `{k}`")
    check(len(p["gpus"]) == 2 and p["gpus"][0]["uuid"] == "GPU-0"
          and p["gpus"][0]["watts"] == 132.3, "per-card watts, name and uuid", str(p["gpus"]))
    check(near(p["gpu_watts"], 138.5, 0.05), "total GPU watts is the sum", str(p["gpu_watts"]))
    check(near(p["dollars_per_hour"], 138.5 / 1000 * p["rate"]["cents_per_kwh"] / 100, 1e-4),
          "$/h = kW x cents / 100", str(p["dollars_per_hour"]))
    check(len(p["days"]) == 7 and p["days"][0]["date"] == p["today"]["date"],
          "seven days, today first", str([d["date"] for d in p["days"]]))
    check(p["today"]["kwh"] is not None and p["today"]["kwh"] > 0,
          "today has measured kWh", str(p["today"]))
    check(p["days"][6]["kwh"] is None and p["days"][6]["measured_seconds"] == 0,
          "a day with nothing measured is None, not zero", str(p["days"][6]))
    check(p["running"] is False, "a sampler whose thread is not running is not `running`")
    check("GPU board power" in p["basis"]["measured"] and "PSU" in p["basis"]["not_measured"],
          "the basis says what is and is not measured")
    stale = power.live(s, now=now + 3600)
    check(stale["gpu_watts"] is None and stale["gpus"] == [],
          "a sample older than STALE_SECONDS is not shown as live draw")
    p = power.live(None)
    check(p["running"] is False and "not running" in (p["last_error"] or ""),
          "no sampler: says so", str(p.get("last_error")))
    json.dumps(power.live(s, now=now))
    check(True, "the payload is JSON-serialisable")


def test_sampler_thread():
    reads = []

    def read():
        reads.append(1)
        return rows(100.0)

    path = os.path.join(_TMP, "thread.json")
    s = power.Sampler(read=read, path=path, interval=0.02)
    s.start()
    time.sleep(0.3)
    check(s.running, "the sampler thread runs")
    s.stop()
    check(not s.running, "and stops")
    check(len(reads) >= 5, "it samples at its interval", str(len(reads)))
    check(os.path.exists(path), "stop() saves the ledger")
    s = power.Sampler(read=lambda: [], path=os.path.join(_TMP, "err.json"))
    s.sample_once()
    check(s.errors == 1 and s.last_error, "an empty nvidia-smi answer is counted as an error")
    s = power.Sampler(read=lambda: (_ for _ in ()).throw(OSError("boom")),
                      path=os.path.join(_TMP, "err2.json"))
    s.sample_once()
    check(s.errors >= 1 and "boom" in (s.last_error or ""), "a raising reader never kills the loop")


def test_vitals_parses_power():
    real = vitals._sh
    try:
        vitals._sh = lambda cmd, timeout=25: (
            "0, NVIDIA GeForce RTX 5060 Ti, 14466, 16311, 99, GPU-de66, 181.18, 180.00\n"
            "1, NVIDIA RTX A4000, 7567, 16376, 0, GPU-43e3, [N/A], 140.00\n")
        g = vitals.gpus()
        check(g[0]["watts"] == 181.18 and g[0]["watts_limit"] == 180.0
              and g[0]["uuid"] == "GPU-de66", "gpus() reads power.draw, power.limit, uuid", str(g[0]))
        check(g[1]["watts"] is None, "[N/A] is None, never 0 W", str(g[1]))
        cmd = []
        vitals._sh = lambda c, timeout=25: cmd.append(c) or ""
        vitals.gpus()
        check("power.draw" in cmd[0][1] and "uuid" in cmd[0][1], "the query asks for them", str(cmd))
        vitals._sh = lambda cmd, timeout=25: "0, NVIDIA RTX A4000, 7000, 16376, 3\n"
        g = vitals.gpus()
        check(g and g[0]["watts"] is None and g[0]["free_mib"] == 9376,
              "a five-column answer still parses", str(g))
    finally:
        vitals._sh = real


def test_power_route():
    import dash_vitals
    code, ctype, body = dash_vitals.handle_get("/dash/api/power")
    d = json.loads(body)
    check(code == 200 and ctype == "application/json" and "rate" in d and "days" in d,
          "/dash/api/power answers with the live() payload", str(code))


def test_livebench_electricity():
    import dash_results
    root = os.path.join(_TMP, "lb")
    run = os.path.join(root, "lb-test")
    os.makedirs(run)
    json.dump({"arms": {"bonsai": {"categories": {
        "coding": {"n": 10, "score": 50, "seconds": {"n": 10, "total": 3600.0}},
        "math": {"n": 10, "score": 50, "seconds": {"n": 10, "total": 3600.0}}}}}},
        open(os.path.join(run, "summary.json"), "w"))
    with open(os.path.join(run, "rows_extra_coding.jsonl"), "w") as f:
        for sec in (100.0, 300.0):
            f.write(json.dumps({"arm": "extra", "status": "ok", "seconds": sec}) + "\n")
        f.write(json.dumps({"arm": "extra", "status": "not_run", "seconds": 999}) + "\n")
    s = dash_results.livebench_summary(root)
    r = s["runs"][0]
    arms = {a["arm"]: a for a in r["arms"]}
    e = arms["bonsai"]["electricity"]
    check(e and e["n"] == 20 and near(e["seconds_per_question"], 360.0),
          "seconds per question from summary totals over every category", str(e))
    check(near(e["wh_per_question"], 138.5 * 360 / 3600, 1e-3),
          "Wh per question = 138.5 W x seconds / 3600", str(e))
    check(near(e["kwh_per_100"], e["wh_per_question"] * 100 / 1000, 1e-4), "kWh per 100 questions")
    check(e["cents_per_kwh"] == [18.435, 24.133]
          and near(e["dollars_per_100"][0], e["kwh_per_100"] * 18.435 / 100, 1e-4),
          "$ per 100 priced at the cheapest and dearest cell", str(e))
    x = arms["extra"]["electricity"]
    check(x and x["n"] == 2 and x["seconds_from"] == "rows" and near(x["seconds_per_question"], 200.0),
          "an arm with no summary seconds falls back to its ok rows", str(x))
    b = r["electricity_basis"]
    check(b and b["estimate"] is True and b["watts"] == 138.5 and "594" in b["evidence"],
          "the run carries the estimate's basis", str(b))


def main() -> int:
    for fn in (test_rate_periods, test_flat_and_extra_overrides, test_integration,
               test_ledger_persists, test_request_energy, test_x_yamadori_carries_energy,
               test_live_payload, test_sampler_thread, test_vitals_parses_power,
               test_power_route, test_livebench_electricity):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    check(not os.path.exists(power.LEDGER_PATH) or power.LEDGER_PATH.startswith(_TMP),
          "no test wrote the real ledger")
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
