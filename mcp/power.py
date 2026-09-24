#!/usr/bin/env python
"""Electricity: what the GPUs draw, what that costs, per day and per request.

WHY THIS EXISTS

The operator owns the hardware, so the only running cost is electricity. This
module turns nvidia-smi's `power.draw` into watts, watt-hours and cents, and
says plainly what it does NOT measure.

WHAT IS MEASURED, AND WHAT IS NOT

  measured      GPU board power per card, nvidia-smi `power.draw`, sampled by
                the proxy process every SAMPLE_SECONDS (the pulse cadence,
                vitals.PULSE_GPU_TTL).
  not measured  CPU, motherboard, RAM, drives, fans, and PSU conversion loss.
                YAMADORI_POWER_EXTRA_WATTS (default 0) adds a flat figure for
                them; every payload carries it as `extra_watts`, separately.

THE PRICE

DTE Energy residential Time of Day 3-7 p.m. standard rate D1.11, rate card in
effect 2025-02-06, EXCLUDING surcharges and taxes. The price of a kWh is
capacity + non-capacity + distribution; RATE_TABLE keeps the three parts so the
sums are checkable against the card. Peak is Monday-Friday 15:00-19:00 local
(the machine's clock: Detroit). Holidays are not modelled: the brief this was
built from does not list any, and the tariff text was not checked for them.

YAMADORI_POWER_RATE_CENTS replaces the table with one all-in flat price per
kWh (for example a bill's total divided by its kWh, surcharges and taxes
included); the period then reads "flat".

THE DAILY LEDGER

Energy is integrated sample to sample (trapezoid) into index/power_ledger.json
(YAMADORI_POWER_LEDGER overrides), one entry per local date, split by rate
period. index/ is derived data and is not in git. A gap longer than
MAX_GAP_SECONDS between samples (the proxy was stopped, the machine slept,
nvidia-smi hung) is NOT integrated: it is counted as `gap_seconds`, because
guessing the draw across it would be inventing a number.

PER REQUEST

request_energy() reads the samples already taken during the request's wall
time -- no nvidia-smi call on the request path. A request shorter than one
sample interval gets the rolling average of the last ROLLING_SECONDS instead,
and says so in `basis`. The whole GPU draw is charged to the request: two
requests in flight at once are each charged the full draw, so their sum
overstates the bill.
"""
from __future__ import annotations

import atexit
import bisect
import datetime as _dt
import json
import os
import threading
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_PATH = os.environ.get(
    "YAMADORI_POWER_LEDGER",
    os.path.abspath(os.path.join(HERE, "..", "index", "power_ledger.json")))

# ---------------------------------------------------------------------------
# The rate card. ONE place; everything else reads it.
# ---------------------------------------------------------------------------

RATE_SOURCE = ("DTE Energy residential Time of Day 3-7 p.m. standard rate "
               "D1.11, rate card in effect 2025-02-06, excluding surcharges "
               "and taxes")
# cents per kWh, by part. distribution is the same in every cell.
RATE_TABLE = {
    "summer": {"months": (6, 7, 8, 9),
               "peak": {"capacity": 5.360, "non_capacity": 9.047,
                        "distribution": 9.726},
               "off_peak": {"capacity": 3.240, "non_capacity": 5.469,
                            "distribution": 9.726}},
    "winter": {"months": (10, 11, 12, 1, 2, 3, 4, 5),
               "peak": {"capacity": 3.839, "non_capacity": 6.480,
                        "distribution": 9.726},
               "off_peak": {"capacity": 3.240, "non_capacity": 5.469,
                            "distribution": 9.726}},
}
PEAK_WEEKDAYS = (0, 1, 2, 3, 4)          # Monday..Friday (datetime.weekday)
PEAK_START_HOUR = 15                     # 15:00 is peak
PEAK_END_HOUR = 19                       # 19:00 is off-peak again

# The measured generating draw, for estimates where no sample exists (the
# benchmark results page). Operator measurement 2026-09-23: 594 one-second
# nvidia-smi power.draw samples while the model was generating. The script
# that took it lived in a scratchpad and is not in the repo (n=1 run).
MEASURED_GENERATING = {
    "watts": 138.5,
    "per_gpu": {"RTX 5060 Ti (generating)": 132.3, "RTX A4000 (idle)": 6.2},
    "samples": 594, "interval_s": 1, "date": "2026-09-23",
    "evidence": "operator measurement, 594 one-second nvidia-smi power.draw "
                "samples while generating; script not in the repo; n=1 run",
}

SAMPLE_SECONDS = 1.0         # vitals.PULSE_GPU_TTL: the pulse cadence
MAX_GAP_SECONDS = 10.0       # a longer gap is counted, not integrated
KEEP_SECONDS = 6 * 3600      # samples kept in memory for per-request energy
ROLLING_SECONDS = 60.0       # the rolling average for short requests
SAVE_EVERY_SECONDS = 60.0    # ledger flush cadence
STALE_SECONDS = 10.0         # a sampler with no sample this old is not live
KEEP_DAYS = 400

# The watts-vs-tokens ring (series()): one entry per sample, memory only.
RING_SAMPLES = 600           # 10 minutes at SAMPLE_SECONDS
# The card the main model runs on: config.yaml pins `bonsai` to it by UUID
# (CUDA_VISIBLE_DEVICES). Main-model tokens are charged to THIS card only.
MAIN_GPU_UUID = os.environ.get("YAMADORI_MAIN_GPU_UUID",
                               "GPU-de660e90-0e9c-d465-b389-6df63021b920")
SWAP_URL = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MAIN_MODEL = os.environ.get("YAMADORI_MODEL", "bonsai")
SLOTS_TIMEOUT = 0.8          # inside the 1 s loop
MIN_R_SAMPLES = 30           # fewer ok samples than this: no coefficient


def _cents(parts: dict) -> float:
    return round(sum(parts.values()), 3)


def flat_rate_cents() -> float | None:
    raw = os.environ.get("YAMADORI_POWER_RATE_CENTS", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v >= 0 else None


def extra_watts() -> float:
    try:
        v = float(os.environ.get("YAMADORI_POWER_EXTRA_WATTS", "0") or 0)
    except ValueError:
        return 0.0
    return max(0.0, v)


def _local(when) -> _dt.datetime:
    """A naive local datetime from an epoch, a datetime, or None (now)."""
    if when is None:
        return _dt.datetime.now()
    if isinstance(when, _dt.datetime):
        return when
    return _dt.datetime.fromtimestamp(float(when))


def season_of(d: _dt.datetime) -> str:
    return next(s for s, v in RATE_TABLE.items() if d.month in v["months"])


def is_peak(d: _dt.datetime) -> bool:
    return (d.weekday() in PEAK_WEEKDAYS
            and PEAK_START_HOUR <= d.hour < PEAK_END_HOUR)


def _next_change(d: _dt.datetime) -> _dt.datetime:
    """When the period next changes: 19:00 today during peak, otherwise the
    next weekday's 15:00."""
    if is_peak(d):
        return d.replace(hour=PEAK_END_HOUR, minute=0, second=0, microsecond=0)
    day = d.replace(hour=PEAK_START_HOUR, minute=0, second=0, microsecond=0)
    if d.weekday() in PEAK_WEEKDAYS and d.hour < PEAK_START_HOUR:
        return day
    day += _dt.timedelta(days=1)
    while day.weekday() not in PEAK_WEEKDAYS:
        day += _dt.timedelta(days=1)
    return day


def rate_at(when=None) -> dict:
    """The price of a kWh at `when` (epoch, naive local datetime, or now)."""
    d = _local(when)
    flat = flat_rate_cents()
    if flat is not None:
        return {"period": "flat", "season": None, "cents_per_kwh": flat,
                "flat": True, "until": None, "until_local": None,
                "components": None,
                "source": "YAMADORI_POWER_RATE_CENTS (all-in flat rate set "
                          "by the operator)"}
    season = season_of(d)
    period = "peak" if is_peak(d) else "off_peak"
    parts = RATE_TABLE[season][period]
    nxt = _next_change(d)
    return {"period": period, "season": season, "cents_per_kwh": _cents(parts),
            "flat": False, "until": time.mktime(nxt.timetuple()),
            "until_local": nxt.strftime("%a %H:%M"),
            "components": dict(parts), "source": RATE_SOURCE}


def rate_range() -> tuple[float, float]:
    """The cheapest and dearest price in the table (or the flat rate twice)."""
    flat = flat_rate_cents()
    if flat is not None:
        return flat, flat
    cells = [_cents(v[p]) for v in RATE_TABLE.values()
             for p in ("peak", "off_peak")]
    return min(cells), max(cells)


def basis() -> dict:
    """What every figure here is, in words a panel can print verbatim."""
    ex = extra_watts()
    return {"measured": "GPU board power, nvidia-smi power.draw, every card",
            "not_measured": "CPU, motherboard, RAM, drives, fans, PSU loss",
            "extra_watts": ex,
            "extra_watts_env": "YAMADORI_POWER_EXTRA_WATTS",
            "rate": (RATE_SOURCE if flat_rate_cents() is None else
                     "YAMADORI_POWER_RATE_CENTS flat rate"),
            "flat_rate_env": "YAMADORI_POWER_RATE_CENTS",
            "holidays": "not modelled",
            "timezone": "machine local time"}


# ---------------------------------------------------------------------------
# The ledger: one entry per local date.
# ---------------------------------------------------------------------------

def _blank_day() -> dict:
    return {"seconds": 0.0, "gap_seconds": 0.0, "gpu_wh": 0.0, "extra_wh": 0.0,
            "wh": {}, "cents": {}}


def load_ledger(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("days"), dict):
            return data
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        # Kept aside, never overwritten: a ledger is the only record.
        try:
            os.replace(path, f"{path}.bad-{int(time.time())}")
        except OSError:
            pass
    return {"version": 1, "days": {}}


def save_ledger(path: str, ledger: dict) -> None:
    days = ledger.get("days") or {}
    for k in sorted(days)[:-KEEP_DAYS]:
        days.pop(k, None)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(ledger, source=RATE_SOURCE, units={
            "wh": "watt-hours", "cents": "US cents", "seconds": "seconds"}),
            f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def day_summary(date: str, day: dict | None) -> dict:
    """One ledger day as the API shows it: kWh and cents, total and by
    period. A day with nothing measured has None, not zero."""
    d = day or _blank_day()
    wh = d.get("wh") or {}
    cents = d.get("cents") or {}
    measured = float(d.get("seconds") or 0)
    none = measured <= 0

    def kwh(x):
        return None if none else round(x / 1000.0, 6)

    return {"date": date,
            "kwh": kwh(sum(wh.values())),
            "cents": None if none else round(sum(cents.values()), 4),
            "kwh_peak": kwh(wh.get("peak", 0.0)),
            "kwh_off_peak": kwh(wh.get("off_peak", 0.0)),
            "kwh_flat": kwh(wh.get("flat", 0.0)),
            "cents_peak": None if none else round(cents.get("peak", 0.0), 4),
            "cents_off_peak": None if none else round(cents.get("off_peak", 0.0), 4),
            "cents_flat": None if none else round(cents.get("flat", 0.0), 4),
            "gpu_kwh": kwh(float(d.get("gpu_wh") or 0)),
            "extra_kwh": kwh(float(d.get("extra_wh") or 0)),
            "measured_seconds": round(measured, 1),
            "gap_seconds": round(float(d.get("gap_seconds") or 0), 1)}


# ---------------------------------------------------------------------------
# The sampler.
# ---------------------------------------------------------------------------

def _read_vitals() -> list[dict]:
    import vitals
    return vitals.gpus(timeout=3)


def total_watts(rows: list[dict]) -> float | None:
    """Sum of every card's draw; None if any card did not report one (a
    partial sum would under-count and look like a measurement)."""
    if not rows:
        return None
    ws = [r.get("watts") for r in rows]
    if any(not isinstance(w, (int, float)) for w in ws):
        return None
    return float(sum(ws))


class Sampler:
    """Reads the cards every `interval` seconds on a daemon thread, keeps
    recent samples for per-request energy, and integrates the ledger.

    `ingest()` is the whole of the arithmetic and takes its clock as an
    argument, so tests drive it without a thread or a GPU."""

    def __init__(self, read=None, path: str | None = None,
                 interval: float = SAMPLE_SECONDS, clock=time.time,
                 read_tokens=None):
        self._read = read or _read_vitals
        # The token-rate reader for the watts-vs-tokens ring (SlotCounter.
        # read). None reads nothing: only start() -- the proxy -- passes one,
        # so a test's Sampler never talks to a model server.
        self._read_tokens = read_tokens
        self._ring: deque[dict] = deque(maxlen=RING_SAMPLES)
        self._idle: dict | None = None
        self.path = path or LEDGER_PATH
        self.interval = interval
        self._clock = clock
        self._lock = threading.Lock()
        self._t: deque[float] = deque()          # sample times, ascending
        self._w: deque[float | None] = deque()   # total GPU watts per sample
        self._prev: tuple[float, float | None] | None = None
        self._rows: list[dict] = []
        self._rows_at = 0.0
        self._saved_at = clock()
        self.ledger = load_ledger(self.path)
        self.errors = 0
        self.last_error: str | None = None
        self.started_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- arithmetic -------------------------------------------------------

    def ingest(self, t: float, rows: list[dict]) -> None:
        w = total_watts(rows)
        ex = extra_watts()
        with self._lock:
            self._rows, self._rows_at = rows, t
            if self._t and t <= self._t[-1]:
                return                               # clock went backwards
            self._t.append(t)
            self._w.append(w)
            while self._t and self._t[0] < t - KEEP_SECONDS:
                self._t.popleft()
                self._w.popleft()
            prev, self._prev = self._prev, (t, w)
            if prev is None:
                return
            dt = t - prev[0]
            mid = prev[0] + dt / 2
            day = self.ledger["days"].setdefault(
                _local(mid).strftime("%Y-%m-%d"), _blank_day())
            if dt > MAX_GAP_SECONDS or prev[1] is None or w is None:
                day["gap_seconds"] = day.get("gap_seconds", 0.0) + dt
                return
            gpu_wh = (prev[1] + w) / 2 * dt / 3600.0
            extra_wh = ex * dt / 3600.0
            rate = rate_at(mid)
            p = rate["period"]
            day["seconds"] = day.get("seconds", 0.0) + dt
            day["gpu_wh"] = day.get("gpu_wh", 0.0) + gpu_wh
            day["extra_wh"] = day.get("extra_wh", 0.0) + extra_wh
            day.setdefault("wh", {})
            day.setdefault("cents", {})
            day["wh"][p] = day["wh"].get(p, 0.0) + gpu_wh + extra_wh
            day["cents"][p] = (day["cents"].get(p, 0.0)
                               + (gpu_wh + extra_wh) / 1000.0 * rate["cents_per_kwh"])

    def watts_between(self, t0: float, t1: float) -> tuple[float | None, int]:
        """Mean total GPU watts over the samples taken in [t0, t1]."""
        with self._lock:
            ts = list(self._t)
            i, j = bisect.bisect_left(ts, t0), bisect.bisect_right(ts, t1)
            ws = [w for w in list(self._w)[i:j] if w is not None]
        return (sum(ws) / len(ws), len(ws)) if ws else (None, 0)

    def rolling(self, seconds: float = ROLLING_SECONDS,
                now: float | None = None) -> tuple[float | None, int]:
        now = self._clock() if now is None else now
        return self.watts_between(now - seconds, now)

    def latest(self) -> tuple[float, list[dict]]:
        with self._lock:
            return self._rows_at, list(self._rows)

    # -- the thread -------------------------------------------------------

    def sample_once(self) -> None:
        try:
            rows = self._read()
        except Exception as e:                                   # noqa: BLE001
            rows = []
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"[:200]
        if not rows:
            self.errors += 1
            self.last_error = self.last_error or "nvidia-smi returned no rows"
        else:
            self.last_error = None
        t = self._clock()
        self.ingest(t, rows)
        if self._read_tokens is not None:
            try:
                tok = self._read_tokens()
            except Exception as e:                               # noqa: BLE001
                tok = {"state": "error", "why": f"{type(e).__name__}: {e}"[:200]}
            self.ring_add(t, rows, tok)

    def ring_add(self, t: float, rows: list[dict], tok: dict | None) -> None:
        """One second of the watts-vs-tokens ring: every card's draw and the
        main model's token deltas (SlotCounter.read) at the same tick.
        Memory only, bounded at RING_SAMPLES."""
        with self._lock:
            self._ring.append({
                "t": t,
                "gpus": [{"index": r.get("index"), "name": r.get("name"),
                          "uuid": r.get("uuid"), "watts": r.get("watts")}
                         for r in rows],
                "tok": dict(tok or {"state": "error", "why": "no reading"})})

    def ring(self) -> list[dict]:
        with self._lock:
            return list(self._ring)

    def save(self) -> None:
        with self._lock:
            snap = json.loads(json.dumps(self.ledger))
            self._saved_at = self._clock()
        try:
            save_ledger(self.path, snap)
        except OSError as e:
            self.last_error = f"ledger not saved: {type(e).__name__}: {e}"[:200]

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = self._clock()
            self.sample_once()
            if self._clock() - self._saved_at >= SAVE_EVERY_SECONDS:
                self.save()
            self._stop.wait(max(0.05, self.interval - (self._clock() - t0)))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.started_at = self._clock()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="power-sampler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.save()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())


_SAMPLER: Sampler | None = None
_START_LOCK = threading.Lock()


def start(**kw) -> Sampler:
    """Start the one sampler in this process. Only mcp/server.py main() calls
    it: importing this module starts nothing. The proxy's sampler also reads
    the main model's token counters (SlotCounter) in the same 1 s loop."""
    global _SAMPLER
    with _START_LOCK:
        if _SAMPLER is None:
            kw.setdefault("read_tokens", SlotCounter().read)
            _SAMPLER = Sampler(**kw)
            atexit.register(_SAMPLER.stop)
        _SAMPLER.start()
        return _SAMPLER


def sampler() -> Sampler | None:
    return _SAMPLER


def fresh_gpus(max_age: float) -> list[dict] | None:
    """The sampler's newest full nvidia-smi rows if younger than `max_age`,
    so the pulse does not start a second nvidia-smi for the same second."""
    s = _SAMPLER
    if s is None:
        return None
    at, rows = s.latest()
    return rows if rows and time.time() - at <= max_age else None


# ---------------------------------------------------------------------------
# What callers read.
# ---------------------------------------------------------------------------

def request_energy(t_start: float | None, t_end: float | None = None,
                   s: Sampler | None = None) -> dict | None:
    """x_yamadori.energy for one request. Never raises; None only when the
    request has no start time. Cheap: reads samples already taken."""
    if not isinstance(t_start, (int, float)):
        return None
    try:
        t_end = time.time() if t_end is None else t_end
        s = s if s is not None else _SAMPLER
        seconds = max(0.0, t_end - t_start)
        rate = rate_at(t_end)
        ex = extra_watts()
        out = {"seconds": round(seconds, 2), "gpu_watts_avg": None,
               "extra_watts": ex, "wh": None, "cents": None,
               "rate_period": rate["period"],
               "cents_per_kwh": rate["cents_per_kwh"],
               "samples": 0, "basis": "no_sampler"}
        if s is None:
            return out
        w, n = s.watts_between(t_start, t_end)
        how = "window"
        if w is None:
            w, n = s.rolling(ROLLING_SECONDS, now=t_end)
            how = "rolling"
        if w is None:
            out["basis"] = "no_samples"
            return out
        wh = (w + ex) * seconds / 3600.0
        out.update(gpu_watts_avg=round(w, 1), wh=round(wh, 3),
                   cents=round(wh / 1000.0 * rate["cents_per_kwh"], 4),
                   samples=n, basis=how)
        return out
    except Exception as e:                                       # noqa: BLE001
        return {"seconds": None, "error": f"{type(e).__name__}: {e}"[:200]}


def live(s: Sampler | None = None, now: float | None = None) -> dict:
    """The dashboard payload (/dash/api/power, and `power` on the vitals
    snapshot and pulse)."""
    s = s if s is not None else _SAMPLER
    now = time.time() if now is None else now
    rate = rate_at(now)
    ex = extra_watts()
    today = _local(now).date()
    dates = [(today - _dt.timedelta(days=i)).isoformat() for i in range(7)]
    out = {"running": False, "at": None, "age_s": None,
           "interval_s": s.interval if s else SAMPLE_SECONDS,
           "gpus": [], "gpu_watts": None, "extra_watts": ex,
           "total_watts": None, "rolling_watts": None,
           "rolling_seconds": ROLLING_SECONDS,
           "rate": rate, "dollars_per_hour": None,
           "today": day_summary(dates[0], None),
           "days": [day_summary(d, None) for d in dates],
           "errors": 0, "last_error": None,
           "measured_generating": MEASURED_GENERATING,
           "basis": basis()}
    if s is None:
        out["last_error"] = ("the power sampler is not running in this "
                             "process (it starts with mcp/server.py)")
        return out
    at, rows = s.latest()
    fresh = bool(rows) and now - at <= STALE_SECONDS
    gw = total_watts(rows) if fresh else None
    roll, _ = s.rolling(ROLLING_SECONDS, now=now)
    with s._lock:
        days = {d: json.loads(json.dumps(s.ledger["days"].get(d)))
                if s.ledger["days"].get(d) else None for d in dates}
    out.update(
        running=s.running and fresh, at=at or None,
        age_s=round(now - at, 1) if at else None,
        gpus=[{"index": r.get("index"), "name": r.get("name"),
               "uuid": r.get("uuid"), "watts": r.get("watts"),
               "watts_limit": r.get("watts_limit")} for r in rows] if fresh else [],
        gpu_watts=None if gw is None else round(gw, 1),
        total_watts=None if gw is None else round(gw + ex, 1),
        rolling_watts=None if roll is None else round(roll, 1),
        dollars_per_hour=(None if gw is None else
                          round((gw + ex) / 1000.0 * rate["cents_per_kwh"] / 100.0, 4)),
        today=day_summary(dates[0], days[dates[0]]),
        days=[day_summary(d, days[d]) for d in dates],
        errors=s.errors, last_error=s.last_error)
    return out


def ledger_days(s: Sampler | None = None, path: str | None = None) -> dict:
    """Every ledger day, {date: raw day}: the running sampler's in-memory
    ledger (fresher than the file, which is flushed every minute), else the
    file read-only. Unlike load_ledger() this never moves a bad file aside:
    a dashboard read must not change the only record."""
    s = s if s is not None else _SAMPLER
    if s is not None:
        with s._lock:
            return json.loads(json.dumps(s.ledger.get("days") or {}))
    try:
        with open(path or LEDGER_PATH, encoding="utf-8") as f:
            data = json.load(f)
        days = data.get("days") if isinstance(data, dict) else None
        return days if isinstance(days, dict) else {}
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Tokens per second, passively: the main model's /slots counters.
#
# WHY /slots. llama-server's /metrics would give monotonic counters, but it
# needs --metrics and config.yaml does not set it (GET /metrics answers 501).
# The proxy's own streamed events miss everything that does not stream
# through it -- deep thinking's hops (model.post), the tools API's
# summarize_text, the worker -- and those run on the same card. /slots sees
# every generation on the server whoever sent it, which is why the watchdog
# already reads it. It is asked DIRECTLY on the port llama-swap reports for
# a loaded model: llama-swap's GET /running lists what is loaded and loads
# nothing, and a request that never goes through llama-swap's /upstream
# cannot wake a model. Not loaded is a state, and /slots is then not called.
#
# WHAT IT UNDERCOUNTS. /slots shows a task's running counters (n_decoded,
# n_prompt_tokens_processed) only while the slot is busy; the tokens a task
# makes between the last read and its end are not seen, so each finished
# generation loses at most one sample interval (~1 s) of tokens. The rate is
# a floor, never an overcount.
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: float):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


class SlotCounter:
    """Token deltas since the previous read, from the main model's /slots."""

    def __init__(self, get=None, clock=time.time, model: str = MAIN_MODEL,
                 swap: str = SWAP_URL):
        self._get = get or _get_json
        self._clock = clock
        self.model = model
        self.swap = swap
        self._url: str | None = None
        self._url_at = 0.0
        self._prev: dict[int, tuple] = {}     # slot id -> (task, decoded, processed)
        self._primed = False
        self._last_t: float | None = None

    def _where(self, now: float) -> tuple[str, str | None]:
        """(state, the model's own address). Asks llama-swap's /running at
        most every 30 s while the model answers, every read otherwise."""
        if self._url and now - self._url_at < 30:
            return "ready", self._url
        try:
            run = self._get(f"{self.swap}/running", 2.0)
        except Exception as e:                                   # noqa: BLE001
            self._url = None
            return "swap_down", f"llama-swap /running: {type(e).__name__}: {e}"[:200]
        for m in (run or {}).get("running") or []:
            if isinstance(m, dict) and m.get("model") == self.model:
                if m.get("state") != "ready" or not m.get("proxy"):
                    self._url = None
                    return str(m.get("state") or "starting"), None
                # llama-swap reports http://localhost:<port>. On this machine
                # `localhost` tries IPv6 first and costs 200-800 ms per call
                # (measured: curl 0.21 s vs 0.002 s for 127.0.0.1), most of
                # a 1 s tick. llama-server listens on 127.0.0.1.
                url = str(m["proxy"]).rstrip("/").replace("://localhost:", "://127.0.0.1:")
                self._url, self._url_at = url, now
                return "ready", self._url
        self._url = None
        return "not_loaded", None

    def read(self) -> dict:
        now = self._clock()
        dt = None if self._last_t is None else now - self._last_t
        self._last_t = now
        state, where = self._where(now)
        if state != "ready":
            self._prev.clear()
            self._primed = False
            return {"state": state, "why": where, "dt": dt}
        try:
            slots = self._get(f"{where}/slots", SLOTS_TIMEOUT)
        except Exception as e:                                   # noqa: BLE001
            self._url = None                 # re-ask /running next time
            self._prev.clear()
            self._primed = False
            return {"state": "no_answer", "dt": dt,
                    "why": f"/slots: {type(e).__name__}: {e}"[:200]}
        return dict(self.deltas(slots if isinstance(slots, list) else []), dt=dt)

    def deltas(self, slots: list[dict]) -> dict:
        """Tokens decoded and prompt tokens processed since the last read,
        summed over slots. The first read after (re)starting only primes:
        a task already running would otherwise count all its past tokens
        as one second's."""
        decoded = prompt = busy = 0
        seen: dict[int, tuple] = {}
        for s in slots:
            if not isinstance(s, dict) or not s.get("is_processing"):
                continue
            busy += 1
            nt = s.get("next_token")
            nt = nt[0] if isinstance(nt, list) and nt else (nt if isinstance(nt, dict) else {})
            sid = int(s.get("id", -1))
            cur = (s.get("id_task"), int(nt.get("n_decoded") or 0),
                   int(s.get("n_prompt_tokens_processed") or 0))
            seen[sid] = cur
            prev = self._prev.get(sid)
            if prev is not None and prev[0] == cur[0]:
                decoded += max(0, cur[1] - prev[1])
                prompt += max(0, cur[2] - prev[2])
            else:
                # A task that started since the last read: all of it is new.
                decoded += cur[1]
                prompt += cur[2]
        primed = self._primed
        self._prev, self._primed = seen, True
        if not primed:
            return {"state": "ok", "busy": busy, "decoded": None, "prompt": None,
                    "why": "first read primes the counters"}
        return {"state": "ok", "busy": busy, "decoded": decoded, "prompt": prompt}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def series_of(ring: list[dict], now: float, window: float = RING_SAMPLES,
              interval: float = SAMPLE_SECONDS, idle_prev: dict | None = None,
              main_uuid: str = MAIN_GPU_UUID) -> dict:
    """The watts-vs-tokens payload from ring entries. Pure: the tests feed it
    a synthetic ring.

      tok/s       decoded / dt and prompt / dt per sample (dt: time since the
                  previous token read)
      idle        mean watts per card over samples where the model answered
                  and no slot was busy and no token moved
      generating  samples where a slot was busy or a token moved
      J/token     sum(main-card W x dt) / sum(tokens) over generating samples
      marginal    sum((main W - idle main W) x dt) / sum(tokens), same samples
      J/gen token and J/prompt token: the same over samples where ONLY that
                  phase moved (a sample with both cannot be split)
      r           Pearson, main-card W against decode tok/s and against
                  prompt tok/s, over every sample where the model answered;
                  n and window stated; association, not cause
    """
    rows = [e for e in ring if now - e["t"] <= window + 0.5]
    order: list[tuple] = []
    for e in rows:
        for g in e["gpus"]:
            k = (g.get("uuid"), g.get("index"), g.get("name"))
            if k not in order:
                order.append(k)
    order.sort(key=lambda k: (k[0] != main_uuid, k[1] if k[1] is not None else 99))
    gpus = [{"uuid": u, "index": i, "name": nm, "main": u == main_uuid} for u, i, nm in order]
    main_i = next((i for i, g in enumerate(gpus) if g["main"]), None)
    t, watts, dec, pre, busy, states = [], [[] for _ in gpus], [], [], [], []
    for e in rows:
        t.append(round(e["t"] - now, 1))
        by = {g.get("uuid"): g.get("watts") for g in e["gpus"]}
        for i, g in enumerate(gpus):
            w = by.get(g["uuid"])
            watts[i].append(round(w, 1) if isinstance(w, (int, float)) else None)
        tok = e["tok"] or {}
        dt = tok.get("dt") or interval
        ok = tok.get("state") == "ok" and tok.get("decoded") is not None
        dec.append(round(tok["decoded"] / dt, 1) if ok else None)
        pre.append(round(tok["prompt"] / dt, 1) if ok else None)
        busy.append(int(tok.get("busy") or 0) if tok.get("state") == "ok" else None)
        states.append(tok.get("state"))

    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return (sum(xs) / len(xs)) if xs else None

    idle_ix = [k for k, e in enumerate(rows) if busy[k] == 0 and dec[k] == 0 and pre[k] == 0]
    idle = None
    if idle_ix:
        idle = {"watts": [None if (m := mean([watts[i][k] for k in idle_ix])) is None
                          else round(m, 1) for i in range(len(gpus))],
                "n": len(idle_ix), "at": now, "from_window": True}
    elif idle_prev:
        idle = dict(idle_prev, from_window=False)
    gen_ix = [k for k, e in enumerate(rows)
              if dec[k] is not None and (busy[k] or dec[k] or pre[k])]
    stats: dict = {"generating_samples": len(gen_ix), "ok_samples": sum(1 for d in dec if d is not None),
                   "samples": len(rows), "window_s": window}
    if main_i is not None:
        wm = watts[main_i]

        def dtk(k):
            return (rows[k]["tok"].get("dt") or interval)

        def jpt(ix, count):
            ix = [k for k in ix if wm[k] is not None]
            toks = sum(count(k) for k in ix)
            if not ix or toks <= 0:
                return None, len(ix), toks
            return sum(wm[k] * dtk(k) for k in ix) / toks, len(ix), toks
        both = lambda k: rows[k]["tok"]["decoded"] + rows[k]["tok"]["prompt"]  # noqa: E731
        j, n_j, toks = jpt(gen_ix, both)
        stats.update(j_per_token=j, j_n=n_j, tokens=toks)
        idle_main = idle["watts"][main_i] if idle and idle.get("watts") else None
        if j is not None and idle_main is not None:
            ix = [k for k in gen_ix if wm[k] is not None]
            stats["marginal_j_per_token"] = sum((wm[k] - idle_main) * dtk(k) for k in ix) / toks
        else:
            stats["marginal_j_per_token"] = None
        dec_only = [k for k in gen_ix if rows[k]["tok"]["decoded"] > 0 and rows[k]["tok"]["prompt"] == 0]
        pre_only = [k for k in gen_ix if rows[k]["tok"]["prompt"] > 0 and rows[k]["tok"]["decoded"] == 0]
        jd, nd, _ = jpt(dec_only, lambda k: rows[k]["tok"]["decoded"])
        jp, np_, _ = jpt(pre_only, lambda k: rows[k]["tok"]["prompt"])
        stats.update(j_per_gen_token=jd, gen_only_samples=nd,
                     j_per_prompt_token=jp, prompt_only_samples=np_)
        gw = [wm[k] for k in gen_ix if wm[k] is not None]
        stats["gen_watts_mean"] = mean(gw)
        stats["gen_decode_tps_mean"] = mean([dec[k] for k in gen_ix])
        stats["gen_prompt_tps_mean"] = mean([pre[k] for k in gen_ix])
        pairs = [k for k in range(len(rows)) if dec[k] is not None and wm[k] is not None]
        rd = _pearson([dec[k] for k in pairs], [wm[k] for k in pairs]) if len(pairs) >= MIN_R_SAMPLES else None
        rp = _pearson([pre[k] for k in pairs], [wm[k] for k in pairs]) if len(pairs) >= MIN_R_SAMPLES else None
        stats.update(r_decode=rd, r_prompt=rp, r_n=len(pairs), r_min=MIN_R_SAMPLES)
    last = rows[-1]["tok"] if rows else {}
    return {"window_s": window, "interval_s": interval, "now": now,
            "model": {"name": MAIN_MODEL, "state": last.get("state"),
                      "why": last.get("why"), "busy": last.get("busy")},
            "gpus": gpus, "main_index": main_i,
            "t": t, "watts": watts, "decode_tps": dec, "prompt_tps": pre,
            "busy": busy, "idle": idle, "stats": stats,
            "source": {"watts": "nvidia-smi power.draw, every card, 1 s (this sampler)",
                       "tokens": "main model llama-server /slots, 1 s, direct to "
                                 "the port llama-swap reports; never loads a model",
                       "undercount": "a generation's last <1 s of tokens is not "
                                     "seen: the rates are a floor"}}


def series(s: Sampler | None = None, now: float | None = None) -> dict:
    """/dash/api/power/series. In-memory only."""
    s = s if s is not None else _SAMPLER
    now = time.time() if now is None else now
    if s is None:
        return {"running": False, "error": "the power sampler is not running in "
                "this process (it starts with mcp/server.py)"}
    if s._read_tokens is None:
        return {"running": s.running, "error": "this sampler reads no token rates"}
    out = series_of(s.ring(), now, interval=s.interval, idle_prev=s._idle)
    if out["idle"] and out["idle"].get("from_window"):
        s._idle = {k: v for k, v in out["idle"].items() if k != "from_window"}
    return dict(out, running=s.running)


def benchmark_estimate(seconds_total: float | None, n: int | None,
                       watts: float | None = None) -> dict | None:
    """Electricity per question and per 100 questions from wall seconds and
    the measured generating draw. An ESTIMATE: nothing was metered during the
    run. Priced at the cheapest and dearest cell of the rate table."""
    if not isinstance(seconds_total, (int, float)) or not n:
        return None
    w = (MEASURED_GENERATING["watts"] if watts is None else watts) + extra_watts()
    lo, hi = rate_range()
    per_q_s = seconds_total / n
    wh_q = w * per_q_s / 3600.0
    return {"n": n, "seconds_per_question": round(per_q_s, 1),
            "watts": round(w, 1),
            "wh_per_question": round(wh_q, 3),
            "kwh_per_100": round(wh_q * 100 / 1000.0, 4),
            "cents_per_question": [round(wh_q / 1000.0 * lo, 4),
                                   round(wh_q / 1000.0 * hi, 4)],
            "dollars_per_100": [round(wh_q * 100 / 1000.0 * lo / 100.0, 4),
                                round(wh_q * 100 / 1000.0 * hi / 100.0, 4)],
            "cents_per_kwh": [lo, hi]}


def benchmark_basis() -> dict:
    lo, hi = rate_range()
    return {"estimate": True,
            "watts": MEASURED_GENERATING["watts"] + extra_watts(),
            "gpu_watts": MEASURED_GENERATING["watts"],
            "extra_watts": extra_watts(),
            "evidence": MEASURED_GENERATING["evidence"],
            "per_gpu": MEASURED_GENERATING["per_gpu"],
            "seconds": "each arm's wall seconds per answered question "
                       "(summary.json seconds.total / n, else the rows)",
            "overlap": "wall seconds charge the whole GPU to each question; "
                       "where requests overlapped (see the run's condition) "
                       "this overstates the energy per question",
            "cents_per_kwh": [lo, hi],
            "rate": (RATE_SOURCE if flat_rate_cents() is None else
                     "YAMADORI_POWER_RATE_CENTS flat rate"),
            "priced": "cheapest and dearest cell of the rate table"}


if __name__ == "__main__":
    # One read-only sample, printed. Writes no ledger (a temp path).
    import tempfile
    s = Sampler(path=os.path.join(tempfile.mkdtemp(), "power_ledger.json"))
    s.sample_once()
    print(json.dumps(live(s), indent=1))
