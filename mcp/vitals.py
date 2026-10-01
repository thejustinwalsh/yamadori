#!/usr/bin/env python
"""What is actually running, and what it is holding.

WHY THIS EXISTS

An hour went into diagnosing a stack that had been quietly broken since 07:21,
and every symptom was visible the whole time to anyone looking at the right
thing:

  two laya_service processes   a duplicate started and never noticed, the same
                               bug as two proxies bound to port 1234
  GPU1 at 54 MiB free          nothing reported it until an allocation failed
  -b 8192 -ub 2048 on a 0.6B   batch buffers copied from the 27B config
  KV not yet allocated         bonsai sat at 6.6 GB resident and grows to
                               ~14.5 GB on first request, so free memory read
                               healthy right up to the moment it was not

None of that needed clever tooling. It needed one page listing processes,
ports and memory that says plainly when two things are doing the same job.

WHAT IT DELIBERATELY REPORTS

Headroom, not usage. A GPU at 85% is fine; a GPU with 2 GB free when the next
allocation is 8 GB is already broken and does not look it. Free memory against
the largest pending allocation is the number that predicts a crash.

Duplicates, loudly. Every incident today traced to a second copy of something:
two proxies, two Laya services, two benchmark runs. A process list that does
not flag duplication is a list nobody reads.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request

# Laya retired 2026-09-24 (operator; docs/E1.md): E1 heads replace it.
SINGLETONS = ("server.py", "proxy.py", "llama-swap.exe")
# The ports of the services scripts/watchdog.ps1 supervises (the job worker
# has no port; its process is in processes()): the proxy, the tools API
# (mcp/tools_api.py TOOLS_API_PORT), SearXNG (docs/SEARCH.md, loopback) and
# llama-swap, plus the main model's llama-server (config.yaml startPort).
TOOLS_API_PORT = int(os.environ.get("TOOLS_API_PORT", "1235"))
SEARCH_URL = (os.environ.get("YAMADORI_SEARCH_URL") or "http://127.0.0.1:8888").rstrip("/")


def _port_of(url: str, default: int) -> int:
    m = re.search(r":(\d+)(?:/|$)", url)
    return int(m.group(1)) if m else default


PORTS = {1234: "proxy", TOOLS_API_PORT: "tools-api",
         _port_of(SEARCH_URL, 8888): "searxng", 11434: "llama-swap",
         10001: "bonsai"}
# What endpoints() probes: (url, name). Each is a liveness route that never
# loads a model or starts work: llama-swap's model list, the tools API's
# /health (the one route it answers without a key, docs/TOOLS-API.md) and
# SearXNG's /healthz (the watchdog's own probe).
SWAP_URL = (os.environ.get("LLAMA_STACK_URL") or "http://127.0.0.1:11434").rstrip("/")
PROBES = ((f"{SWAP_URL}/v1/models", "llama-swap"),
          (f"http://127.0.0.1:{TOOLS_API_PORT}/health", "tools-api"),
          (f"{SEARCH_URL}/healthz", "searxng"))
PROBE_TIMEOUT = 4
# A card shows red ("tight") below ITS OWN free-VRAM floor, read live, never
# a number the dashboard has to be retuned for (operator, 2026-09-25: "the
# dashboard should be aware of live membudgets"). The floor is the same
# target the card is sized to:
#   main card   YAMADORI_MAIN_FREE_TARGET_MIB, default 600 -- the peak target
#               `-c` is sized against in config.yaml (2026-09-25: 600 MB)
#   others      gpu_room.HEADROOM_MIB (YAMADORI_A4000_HEADROOM_MIB, 1,331) --
#               what the A4000 coordinator keeps free
# Each row carries `floor_mib`, so the UI reads it instead of knowing it.
# History: a fixed TIGHT_MIB of 2048, then 1280 (2026-09-23).
MAIN_FREE_TARGET_MIB = 600


def floor_mib(main: bool) -> int:
    """The free-VRAM floor for a card, read per call."""
    if main:
        try:
            return int(os.environ.get("YAMADORI_MAIN_FREE_TARGET_MIB", "")
                       or MAIN_FREE_TARGET_MIB)
        except ValueError:
            return MAIN_FREE_TARGET_MIB
    try:
        import gpu_room
        return int(gpu_room.HEADROOM_MIB)
    except Exception:                                            # noqa: BLE001
        return 1331


def _sh(cmd: list[str], timeout: int = 25) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:                                            # noqa: BLE001
        return ""


def _float_or_none(x: str) -> float | None:
    """nvidia-smi prints "[N/A]" or "[Not Supported]" for a field a card
    cannot report; that is None, never 0 W."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def gpus(timeout: int = 25) -> list[dict]:
    """One row per card. `watts` is power.draw (board power, W) and
    `watts_limit` power.limit; mcp/power.py integrates `watts` into energy.
    The three power-era fields are appended after the original five, so a
    five-column answer still parses (and reports watts None)."""
    # temperature.gpu (2026-09-30, the PERFORMANCE page's history) is
    # appended last, as the power fields were, so a shorter answer still
    # parses (temp_c None).
    out = _sh(["nvidia-smi",
               "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
               "uuid,power.draw,power.limit,temperature.gpu",
               "--format=csv,noheader,nounits"], timeout=timeout)
    rows = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 5:
            continue
        used, total = int(p[2]), int(p[3])
        rows.append({"index": int(p[0]), "name": p[1], "used_mib": used,
                     "total_mib": total, "free_mib": total - used,
                     "pct": round(100 * used / max(total, 1)),
                     "util": int(p[4] or 0),
                     "uuid": p[5] if len(p) > 5 and p[5] else None,
                     "watts": _float_or_none(p[6]) if len(p) > 6 else None,
                     "watts_limit": _float_or_none(p[7]) if len(p) > 7 else None,
                     "temp_c": _float_or_none(p[8]) if len(p) > 8 else None})
    # The card the main model runs on, by UUID (config.yaml pins `bonsai` by
    # UUID; mcp/power.py uses the same constant). Index order is PCI order and
    # would silently name the wrong card if the cards ever changed slots.
    main_uuid = os.environ.get("YAMADORI_MAIN_GPU_UUID",
                               "GPU-de660e90-0e9c-d465-b389-6df63021b920")
    for r in rows:
        r["main"] = r.get("uuid") == main_uuid
        r["floor_mib"] = floor_mib(r["main"])
        r["tight"] = r["free_mib"] < r["floor_mib"]
    return rows


def power_live() -> dict | None:
    """mcp/power.py live(): watts, the rate period, today's and the last 7
    days' kWh and cents. Real only inside the proxy process, where the
    sampler runs; elsewhere it says the sampler is not running."""
    try:
        import power
        return power.live()
    except Exception as e:                                       # noqa: BLE001
        return {"running": False, "last_error": f"{type(e).__name__}: {e}"[:200]}


def _sampled_gpus(max_age: float) -> list[dict] | None:
    """The power sampler's newest nvidia-smi rows, if fresh: the sampler
    already reads the cards every second in the proxy, so the pulse reuses
    that read instead of starting a second nvidia-smi."""
    try:
        import power
        return power.fresh_gpus(max_age)
    except Exception:                                            # noqa: BLE001
        return None


def processes() -> list[dict]:
    ps = _sh(["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_Process | Where-Object { $_.Name -match "
              "'python|llama-server|llama-swap' } | Select-Object ProcessId,"
              "ParentProcessId,Name,CommandLine,CreationDate | "
              "ConvertTo-Json -Compress"])
    try:
        data = json.loads(ps) if ps.strip() else []
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    out = []
    for p in data:
        cmd = p.get("CommandLine") or ""
        gguf = re.search(r"([A-Za-z0-9._-]+\.gguf)", cmd)
        script = re.search(r"([a-z_]+\.py)", cmd)
        port = re.search(r"--port (\d+)", cmd) or re.search(r":(\d+)\s*$", cmd)
        out.append({"pid": p.get("ProcessId"),
                    "ppid": p.get("ParentProcessId"),
                    "what": (gguf.group(1) if gguf else
                             (script.group(1) if script else p.get("Name"))),
                    "port": port.group(1) if port else "",
                    "started": _started(p.get("CreationDate"))})
    return sorted(out, key=lambda x: str(x["what"]))


def _started(v) -> str:
    """A process's start time as local ISO seconds. Windows PowerShell 5.1's
    ConvertTo-Json writes a DateTime as "/Date(<epoch ms>)/", which the old
    [:19] cut left as "/Date(1790680397077" on the page."""
    s = str(v or "")
    m = re.match(r"^/Date\((-?\d+)(?:[+-]\d{4})?\)/$", s)
    if m:
        try:
            return time.strftime("%Y-%m-%dT%H:%M:%S",
                                 time.localtime(int(m.group(1)) / 1000))
        except (OverflowError, OSError, ValueError):
            return ""
    return s[:19]


def duplicates(procs: list[dict]) -> list[dict]:
    """Independent TREES of a thing that should run once, not raw process count.

    The first version counted processes and immediately lied: it reported two
    copies of laya_service, the orphan was killed, and both died -- because
    they were a parent and its child. laya_service legitimately runs several
    processes, serving HTTP on 1237 and WebSocket on 1238.

    A monitor that cries wolf is worse than no monitor, because it teaches you
    to ignore it, and acting on that first false positive took the decision
    model offline for twenty minutes.

    Grouping by process tree was the second wrong answer. laya_service.py is a
    single process by design -- HTTP on a thread, WebSocket on asyncio, one
    __main__ guard, no multiprocessing -- and the extra entries were launcher
    and shell wrappers left by repeated `nohup` starts. Counting them at all
    was measuring the wrong thing twice.

    PORTS ARE THE TRUTH. One listener on 1237 is one service, whatever the
    process table looks like: verified, pid 35052 owns both 1237 and 1238.
    So process trees are reported for context and `listeners()` decides
    whether anything is actually wrong.
    """
    by_pid = {p["pid"]: p for p in procs if p.get("pid")}

    def root(p: dict) -> int:
        seen_ids = set()
        cur = p
        while True:
            parent = by_pid.get(cur.get("ppid"))
            # Stop at the first ancestor outside the tracked set, and guard
            # against a cycle rather than trusting the table.
            if parent is None or parent["pid"] in seen_ids:
                return cur["pid"]
            seen_ids.add(cur["pid"])
            cur = parent

    trees: dict[str, set] = {}
    for p in procs:
        w = str(p.get("what") or "")
        if w in SINGLETONS:
            trees.setdefault(w, set()).add(root(p))
    return [{"what": k, "pids": sorted(v)} for k, v in trees.items()
            if len(v) > 1]


def listeners() -> list[dict]:
    rows = []
    for line in _sh(["netstat", "-ano"]).splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            port = int(parts[1].rsplit(":", 1)[-1])
        except ValueError:
            continue
        if port in PORTS:
            rows.append({"port": port, "role": PORTS[port],
                         "addr": parts[1], "pid": parts[-1]})
    counts: dict[int, int] = {}
    for r in rows:
        counts[r["port"]] = counts.get(r["port"], 0) + 1
    for r in rows:
        # Two listeners on one port is what served a benchmark from two
        # different builds of the proxy at random.
        r["conflict"] = counts[r["port"]] > 1
    return rows


def _probe(url: str, name: str) -> dict:
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT) as r:
            ok, code = r.status == 200, r.status
    except urllib.error.HTTPError as e:
        ok, code = False, e.code
    except Exception:                                            # noqa: BLE001
        ok, code = False, 0
    return {"name": name, "ok": ok, "code": code,
            "ms": round((time.time() - t0) * 1000)}


def endpoints() -> list[dict]:
    """One row per PROBES entry, probed in parallel so a service that is
    down costs the snapshot one PROBE_TIMEOUT, not one per service."""
    out: list[dict | None] = [None] * len(PROBES)

    def run(i: int, url: str, name: str) -> None:
        out[i] = _probe(url, name)
    ts = [threading.Thread(target=run, args=(i, u, n), daemon=True)
          for i, (u, n) in enumerate(PROBES)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(PROBE_TIMEOUT + 1)
    return [r if r is not None else {"name": PROBES[i][1], "ok": False,
                                     "code": 0, "ms": (PROBE_TIMEOUT + 1) * 1000}
            for i, r in enumerate(out)]


def _with_line(ctx: dict) -> dict:
    """budget.budgets() plus the tiered cache's VRAM line as the model
    server reported it (/props `kv_vram_cells`, engine patch 0041: 0 when
    the pool is not tiered, None when the server does not say). Read from
    mcp/budget.py's own cache of that /props answer; nothing is asked."""
    try:
        import budget
        line = getattr(budget, "_LINE", None)
    except Exception:                                            # noqa: BLE001
        line = None
    if isinstance(ctx, dict) and "error" not in ctx:
        ctx = dict(ctx, vram_line=line if isinstance(line, int) else None)
    return ctx


def main_loaded() -> bool | None:
    """Is the main model loaded, per llama-swap's GET /running (which never
    loads anything)? True / False, None when /running cannot be read."""
    rows = running_rows()
    if rows is None:
        return None
    return any(str(r.get("model")) == MAIN_MODEL and _live_row(r)
               for r in rows)


def cached_context(how: str = "cached: the pool this process last read"
                   ) -> dict:
    """budget.budgets() over the pool this process last READ; asks nothing.
    An explained unknown when no pool was ever read."""
    try:
        import budget
        pool = budget.known_pool()
        if pool is None:
            return {"error": f"the pool is not known yet: {MAIN_MODEL} has not "
                             "been read by this process and is not loaded "
                             "(a view never loads a model)",
                    "pool_read": how}
        return dict(_with_line(budget.budgets(pool)), pool_read=how)
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def context_pool() -> dict:
    """budget.budgets() for the dashboard. A VIEW NEVER LOADS A MODEL
    (2026-09-30): the pool is re-read only from the main model's own server
    (budget.refresh_direct: never llama-swap's /upstream) and only while
    llama-swap's /running lists it; otherwise the pool this process last
    read, or an explained unknown. `pool_read` says which."""
    try:
        import budget
        loaded = main_loaded()
        if loaded:
            _, how = budget.refresh_direct()
        elif loaded is None:
            how = ("cached: llama-swap /running could not be read, so nothing "
                   "was asked (a view never loads a model)")
        else:
            how = (f"cached: {MAIN_MODEL} is not loaded (llama-swap /running), "
                   "so nothing was asked")
        return cached_context(how)
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# PULSE: what the model is doing this second.
#
# snapshot() costs ~0.5 s, most of it the PowerShell process listing, which is
# why the page polls it every 5 s. The tokonoma animates from state that
# changes faster than that: which slots are decoding and how fast, which tool
# just ran, whether a request just arrived. pulse() is that subset, and every
# read in it is bounded:
#
#   slots()   llama-server /slots, 1 s timeout, last good answer kept
#   gpus()    nvidia-smi, cached for PULSE_GPU_TTL, 3 s timeout; in the
#             proxy, the power sampler's own 1 s read is reused instead
#   power     mcp/power.py live(): in-memory samples and the ledger, no I/O
#   tools()   the corpus log, read-only, 0.5 s busy timeout, newest rows only
#   queue()   job counts, read-only
#   lanes()   admission's in-process counters (only real inside the proxy)
#
# Read-only throughout: nothing here writes a file, a row or a request.
# ---------------------------------------------------------------------------

MAIN_MODEL = os.environ.get("YAMADORI_MODEL", "bonsai")
SLOTS_URL = os.environ.get("YAMADORI_MODEL_SERVER", "http://127.0.0.1:10001") + "/slots"
SLOTS_TIMEOUT = 1.0
PULSE_GPU_TTL = 1.0
TOOL_WINDOW = 600          # seconds of tool calls counted in `calls_window`
_lock = threading.Lock()
_slot_prev: dict[int, tuple] = {}     # id -> (task, decoded, processed, t)
_slot_rate: dict[int, tuple] = {}     # id -> (decode tok/s, prefill tok/s)
_gpu_cache: tuple[float, list] = (0.0, [])
_strata_cache: tuple[float, dict | None] = (0.0, None)


def _slot_row(s: dict, now: float) -> dict:
    nt = (s.get("next_token") or [{}])
    nt = nt[0] if isinstance(nt, list) and nt else (nt if isinstance(nt, dict) else {})
    sid = int(s.get("id", -1))
    busy = bool(s.get("is_processing"))
    decoded = int(nt.get("n_decoded") or 0)
    processed = int(s.get("n_prompt_tokens_processed") or 0)
    prompt = int(s.get("n_prompt_tokens") or 0)
    task = s.get("id_task")
    state = "idle" if not busy else ("prefill" if decoded == 0 else "decode")
    prev = _slot_prev.get(sid)
    tps, pps = _slot_rate.get(sid, (0.0, 0.0))
    if not busy:
        tps, pps = 0.0, 0.0
        _slot_prev.pop(sid, None)
    elif prev is None or prev[0] != task:
        _slot_prev[sid] = (task, decoded, processed, now)
        tps, pps = 0.0, 0.0
    elif now - prev[3] >= 0.4:
        dt = now - prev[3]
        tps = max(0.0, (decoded - prev[1]) / dt)
        pps = max(0.0, (processed - prev[2]) / dt)
        _slot_prev[sid] = (task, decoded, processed, now)
    _slot_rate[sid] = (tps, pps)
    return {"id": sid, "state": state, "n_ctx": int(s.get("n_ctx") or 0),
            # n_prompt_tokens grows with every decoded token on this build, so
            # it is already the context the slot holds.
            "ctx": max(prompt, processed + decoded),
            "prompt": prompt, "processed": processed, "decoded": decoded,
            "remain": int(nt.get("n_remain") or 0) if busy else 0,
            "tps": round(tps, 1), "pps": round(pps, 1)}


RUNNING_TTL_S = 2.0
_running_cache: tuple[float, list | None] = (-1e9, None)


def running_rows(max_age: float = RUNNING_TTL_S) -> list[dict] | None:
    """llama-swap's GET /running rows (gpu_room.running: it reports, it
    never loads anything), at most one read per `max_age` seconds. None when
    it cannot be read."""
    global _running_cache
    now = time.time()
    with _lock:
        t, rows = _running_cache
        if now - t < max_age:
            return rows
    try:
        import gpu_room
        rows = gpu_room.running(gpu_room.default_upstream())
    except Exception:                                            # noqa: BLE001
        rows = None
    with _lock:
        _running_cache = (now, rows)
    return rows


def _gguf(cmd: str) -> str | None:
    """The model file a llama-swap row runs: the basename after -m/--model."""
    m = re.search(r"(?:^|\s)(?:-m|--model)\s+\"?([^\s\"]+\.gguf)", cmd or "")
    return re.split(r"[\\/]", m.group(1))[-1] if m else None


def _live_row(r: dict) -> bool:
    return str(r.get("state") or "ready") != "stopped"


def _max_mode():
    """mcp/max_mode.py, or None where it cannot be imported."""
    try:
        import max_mode
        return max_mode
    except Exception:                                            # noqa: BLE001
        return None


def serving() -> dict:
    """Which main model holds the main card (docs/FLASH-NEXT.md,
    mcp/max_mode.py): tier `max` is served by the max model when
    YAMADORI_MAX_MODEL is set, every other tier by the main model, one of
    them loaded at a time. `loaded` is llama-swap's /running (model, state,
    port, gguf), None when it could not be read; the in-flight counts and
    the switch are max_mode's own, real only in the proxy process."""
    rows = running_rows()
    loaded = None if rows is None else [
        {"model": str(r.get("model")), "state": str(r.get("state") or "ready"),
         "port": _port_of(str(r.get("proxy") or ""), 0) or None,
         "gguf": _gguf(str(r.get("cmd") or ""))} for r in rows]
    mm = _max_mode()
    snap: dict = {}
    active = False
    err = None
    if mm is not None:
        try:
            snap = mm.snapshot()
            active = bool(snap.get("enabled")) and bool(mm.max_active())
        except Exception as e:                                   # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:200]
    main = str(snap.get("main") or MAIN_MODEL)
    mx = str(snap.get("max") or "") or None
    names = {x["model"] for x in loaded or [] if x["state"] != "stopped"}
    on_card = (mx if mx and mx in names else main if main in names else None)
    return {"enabled": bool(snap.get("enabled")), "main": main, "max": mx,
            "max_tier": getattr(mm, "MAX_TIER", "max") if mm else "max",
            "on_card": on_card, "max_active": active,
            "inflight": snap.get("inflight") or {},
            "switching_to": snap.get("switching_to"),
            "last_max_end": snap.get("last_max_end") or None,
            "idle_s": snap.get("idle_s"), "loaded": loaded, "error": err}


def _slots_target() -> tuple[str, str]:
    """(the /slots URL, the model it belongs to): the main model's server,
    or another tier model's while it holds the card (the table's MODELS:
    flash-next, mirai-s) -- llama-swap gives it its own port, read from GET
    /running (which never loads), so SLOTS_URL would only say the main model
    is not loaded. Never /upstream/<model>/: that would load an unloaded
    one."""
    mm = _max_mode()
    if mm is not None and getattr(mm, "ENABLED", False):
        others = {m for m in list(getattr(mm, "MODELS", None) or []) + [getattr(mm, "MAX", "")]
                  if m and m != MAIN_MODEL}
        for r in running_rows() or []:
            if str(r.get("model")) in others and _live_row(r) and r.get("proxy"):
                return str(r["proxy"]).rstrip("/") + "/slots", str(r.get("model"))
    return SLOTS_URL, MAIN_MODEL


def off_card_why() -> str | None:
    """Why the main model is not answering, when the reason is known and not
    an outage: a bench window holds the card (the gpu lane's pause record,
    jobs.pause: its `why`), max mode's model holds it, or llama-swap has the
    model unloaded (its GET /running, which never loads anything). None when
    none says so. The dashboard printed "/slots not answering . URLError:
    timed out" through the Flash-Next gate (2026-09-28) with nothing saying
    the gate held the card."""
    held = None
    try:
        import jobs
        rec = jobs.paused("gpu")
        if rec:
            held = str(rec.get("why") or rec.get("by") or "a bench window")
    except Exception:                                            # noqa: BLE001
        held = None
    rows = running_rows()
    loaded = None if rows is None else any(
        str(r.get("model")) == MAIN_MODEL and _live_row(r) for r in rows)
    mm = _max_mode()
    mx = getattr(mm, "MAX", "") if mm is not None and getattr(mm, "ENABLED", False) else ""
    if mx and rows is not None and loaded is False and any(
            str(r.get("model")) == mx and _live_row(r) for r in rows):
        return f"{MAIN_MODEL} is off the card: max mode ({mx}) holds it"
    if held and loaded is not True:
        return f"{MAIN_MODEL} is off the card: {held} holds it"
    if loaded is False:
        return f"{MAIN_MODEL} is not loaded (llama-swap loads it on the next request)"
    return None


def _slot_roles(rows: list[dict]) -> None:
    """Each slot's role in the slot layout (mcp/slots.py: conversations on
    0..n-2, the CHILD slot n-1 shared by deep thinking, the decider, side
    calls and an as-sent compaction), and -- in the proxy process, where
    the pins live -- whether a conversation is pinned to it and whether it
    is the primary conversation's (slots RANKS). Read-only: child_slot()
    and snapshot() are mcp/slots.py's own accessors."""
    try:
        import slots as slot_map
        child = slot_map.child_slot(len(rows))
        snap = slot_map.snapshot()
    except Exception:                                            # noqa: BLE001
        return
    pins = snap.get("pins") or {}
    pinned = set(pins.values())
    primary = (snap.get("ranks") or {}).get("primary")
    primary_slot = pins.get(primary) if primary else None
    for r in rows:
        r["role"] = "child" if r["id"] == child else "conversation"
        r["pinned"] = r["id"] in pinned
        r["primary"] = primary_slot is not None and r["id"] == primary_slot


def slots(url: str | None = None) -> dict:
    """Per-slot state from llama-server: idle / prefill / decode, context
    held, and decode and prefill tokens per second (from the change in
    n_decoded since the last read). {"ok": False, "error": ...} when the
    server does not answer within SLOTS_TIMEOUT -- a model swapped out by
    llama-swap is not listening, which is a state, not an outage. `model`
    names whose server answered (the max model's while it holds the card)."""
    t0 = time.time()
    mm = _max_mode()
    target, model = (url, None) if url else _slots_target()
    try:
        with urllib.request.urlopen(target, timeout=SLOTS_TIMEOUT) as r:
            data = json.load(r)
    except Exception as e:                                       # noqa: BLE001
        why = off_card_why()
        return {"ok": False,
                "error": (why or f"{type(e).__name__}: {e}")[:200],
                "cause": f"{type(e).__name__}: {e}"[:200], "off_card": bool(why),
                "model": model, "slots": [],
                "ms": round((time.time() - t0) * 1000)}
    if not isinstance(data, list):
        return {"ok": False, "error": "unexpected /slots shape", "slots": [],
                "model": model, "ms": round((time.time() - t0) * 1000)}
    now = time.time()
    with _lock:
        rows = [_slot_row(s, now) for s in data if isinstance(s, dict)]
    _slot_roles(rows)
    # A LOCKED model's slots are READ (coordinator, 2026-09-30: a cheap /slots read of a loaded model does not
    # compete for the card); the flag says the card runs its one conversation alone (the tier table)
    locked = bool(model and mm is not None and getattr(mm, "ENABLED", False) and getattr(mm, "TABLE", None) is not None
                  and mm.TABLE.locked(model))
    return {"ok": True, "slots": rows, "ms": round((now - t0) * 1000),
            "model": model, "locked": locked,
            "decoding": sum(1 for s in rows if s["state"] == "decode"),
            "prefilling": sum(1 for s in rows if s["state"] == "prefill"),
            "tps": round(sum(s["tps"] for s in rows), 1)}


def lanes() -> dict | None:
    """Admission lanes held right now (main and helper), from the proxy's own
    counters. Only meaningful inside the proxy process: anywhere else these
    are a fresh module's zeros, so `in_proxy` says which it is."""
    try:
        import sys
        import admission
        snap = admission.snapshot()
    except Exception:                                            # noqa: BLE001
        return None
    inflight = snap.get("inflight") or {}
    helper = snap.get("helper") or {}
    return {"main": int(inflight.get("main") or 0),
            # Investigations hold the threading lane, not the asyncio one.
            "helper": int(inflight.get("helper") or 0) + int(helper.get("inflight") or 0),
            "main_lanes": snap.get("main_lanes"), "helper_lanes": snap.get("helper_lanes"),
            "admitted": snap.get("admitted"), "queued": snap.get("queued"),
            "refused": snap.get("refused"),
            "in_proxy": "server" in sys.modules or "proxy" in sys.modules}


def _ro(path: str) -> sqlite3.Connection:
    uri = pathlib.Path(path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=0.5)
    con.execute("PRAGMA busy_timeout=500")
    return con


def tools(db: str | None = None, window: float = TOOL_WINDOW) -> dict | None:
    """The newest tool calls, from the corpus log: the last one (and whether it
    is still running), calls and requests in the last `window` seconds, and
    the few most recent with their latency. Newest rows only (by rowid), so
    the read stays constant-time as the log grows. None if unreadable."""
    try:
        import corpus
        con = _ro(db or corpus.CORPUS_DB)
    except Exception:                                            # noqa: BLE001
        return None
    now = time.time()
    try:
        top = con.execute("SELECT MAX(id) FROM events").fetchone()[0] or 0
        rows = con.execute(
            "SELECT id, turn, ts, kind, name, "
            "CASE WHEN kind='tool_result' THEN json_extract(payload,'$.ms') END, "
            "CASE WHEN kind='tool_result' THEN json_extract(payload,'$.empty') END "
            "FROM events WHERE id > ? ORDER BY id DESC LIMIT 400",
            (top - 400,)).fetchall()
        counts = dict(con.execute(
            "SELECT kind, COUNT(*) FROM events WHERE id > ? AND ts >= ? "
            "GROUP BY kind", (top - 20000, now - window)).fetchall())
    except Exception:                                            # noqa: BLE001
        return None
    finally:
        con.close()
    recent: list[dict] = []
    open_call: dict | None = None
    finished: set[tuple] = set()
    last_turn = None
    answered: set[str] = set()
    for rid, turn, ts, kind, name, ms, empty in rows:          # newest first
        if kind == "answer":
            answered.add(turn)
        if kind == "turn" and last_turn is None:
            last_turn = {"id": rid, "at": ts, "open": turn not in answered
                         and now - ts < 900}
        if kind == "tool_result":
            finished.add((turn, name))
            if len(recent) < 8:
                recent.append({"id": rid, "name": name, "at": ts,
                               "ms": ms, "empty": bool(empty)})
        elif kind == "tool_call" and open_call is None and not recent:
            # The newest tool event is a call with no result yet.
            if (turn, name) not in finished and now - ts < 300:
                open_call = {"id": rid, "name": name, "at": ts}
    calls = [r for r in rows if r[3] == "tool_call"]
    last = calls[0] if calls else None
    return {"last": ({"id": last[0], "name": last[4], "at": last[2]}
                     if last else None),
            "running": open_call,
            "recent": recent,
            "calls_window": int(counts.get("tool_call", 0)),
            "turns_window": int(counts.get("turn", 0)),
            "answers_window": int(counts.get("answer", 0)),
            "window_seconds": window,
            "last_turn": last_turn}


def queue(db: str | None = None) -> dict | None:
    """Job counts by state, and the oldest queued job's age. Read-only."""
    try:
        import jobs
        con = _ro(db or jobs.DB)
    except Exception:                                            # noqa: BLE001
        return None
    try:
        by = {s: 0 for s in ("queued", "running", "done", "errored",
                             "cancelled")}
        for s, n in con.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state"):
            by[s] = n
        oldest = con.execute(
            "SELECT MIN(created) FROM jobs WHERE state='queued'").fetchone()[0]
    except Exception:                                            # noqa: BLE001
        return None
    finally:
        con.close()
    return {"states": by,
            "oldest_queued_age": round(time.time() - oldest) if oldest else None}


def pulse() -> dict:
    """The fast subset the tokonoma animates from. See the block above."""
    global _gpu_cache, _strata_cache
    now = time.time()
    with _lock:
        fresh_gpu = now - _gpu_cache[0] < PULSE_GPU_TTL
        fresh_strata = now - _strata_cache[0] < 10
    if not fresh_gpu:
        g = _sampled_gpus(2 * PULSE_GPU_TTL)
        if g is None:
            g = gpus(timeout=3)
        with _lock:
            _gpu_cache = (now, g)
    if not fresh_strata:
        st = strata()
        with _lock:
            _strata_cache = (now, st)
    # The cached pool only: no request at all (snapshot() re-reads it, and
    # only from the main model's own server while it is loaded).
    ctx = cached_context()
    return {"at": now, "gpus": _gpu_cache[1], "slots": slots(),
            "lanes": lanes(), "tools": tools(), "queue": queue(),
            "seed": seed(), "strata": _strata_cache[1], "context": ctx,
            "power": power_live()}


def snapshot() -> dict:
    g = _sampled_gpus(2 * PULSE_GPU_TTL)
    procs, g, lis = processes(), (g if g is not None else gpus()), listeners()
    dups, eps = duplicates(procs), endpoints()
    warnings = []
    for x in g:
        if x["tight"]:
            warnings.append(f"GPU{x['index']} has only {x['free_mib']} MiB "
                            f"free (under its {x.get('floor_mib')} MiB floor); a large "
                            f"prefill spike may not fit")
    for d in dups:
        # Context, not an alarm: a second tree is only a fault if it has
        # also taken a port, which `listeners()` reports separately.
        pass
    for r in lis:
        if r["conflict"]:
            warnings.append(f"more than one listener on port {r['port']}; "
                            f"requests are answered at random")
    for e in eps:
        if not e["ok"]:
            warnings.append(f"{e['name']} is not answering")
    return {"at": int(time.time()), "gpus": g, "processes": procs,
            "listeners": lis, "duplicates": dups, "endpoints": eps,
            "context": context_pool(), "seed": seed(), "strata": strata(),
            "slots": slots(), "lanes": lanes(), "tools": tools(),
            "queue": queue(), "power": power_live(), "tree": tree(),
            "serving": serving(),
            "warnings": warnings}


def tree() -> dict | None:
    """What the tokonoma's roots, moss and foliage read (mcp/tree_sources.py):
    index breadth and freshness, cached a minute, and the last requests'
    fan-out and recall (in memory, proxy process only). None if unreadable."""
    try:
        import tree_sources
        return tree_sources.snapshot()
    except Exception:                                            # noqa: BLE001
        return None


def strata() -> dict | None:
    """Search result tiers over the last 24 h: TAPROOT (a declaration with the
    name), BRANCH (both retrievers agree), SHOOT (one retriever only), and how
    many searches reported tiers. Counted at logging time by
    corpus.result_strata. None only if the corpus cannot be read."""
    try:
        import corpus
        return corpus.strata_totals()
    except Exception:                                            # noqa: BLE001
        return None


def seed() -> dict | None:
    """The concept seed most recently put in a prompt, or None if none ever
    was. Read from the file concept_seed.record() writes, so it is the same
    answer whichever process asks."""
    try:
        import concept_seed
        return concept_seed.last()
    except Exception:                                            # noqa: BLE001
        return None


if __name__ == "__main__":
    s = snapshot()
    for x in s["gpus"]:
        flag = "   <-- TIGHT" if x["tight"] else ""
        print(f"  GPU{x['index']} {x['name'][:20]:<20} "
              f"{x['used_mib']:>6}/{x['total_mib']} MiB ({x['pct']:>3}%)  "
              f"{x['free_mib']:>6} free{flag}")
    c = s["context"]
    if "pool" in c:
        print(f"\n  pool {c['pool']}  main {c['main']}  helper {c['helper']}  "
              f"reserve {c['reserve']}  ({c['gib']} GiB of KV)")
    print(f"\n  {len(s['processes'])} processes")
    for p in s["processes"]:
        port = f"port {p['port']}" if p["port"] else ""
        print(f"    {str(p['pid']):<7} {str(p['what'])[:42]:<42} {port}")
    print()
    for w in s["warnings"]:
        print(f"  WARNING  {w}")
    if not s["warnings"]:
        print("  no warnings")
