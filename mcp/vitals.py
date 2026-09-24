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
import urllib.request

SINGLETONS = ("laya_service.py", "server.py", "proxy.py", "llama-swap.exe")
PORTS = {1234: "proxy", 11434: "llama-swap", 1237: "laya", 10001: "bonsai"}
# Below this much free VRAM a card shows red ("tight"). 1280 MiB, the
# operator's call on 2026-09-23: the old 2048 was the floor for a card that
# also drove the Windows display, whose apps take VRAM without asking. The
# display now runs on the iGPU, and the MTP build at -c 163840 idles at
# ~1.7 GB free, so 2048 painted a healthy card red.
TIGHT_MIB = 1280


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
    out = _sh(["nvidia-smi",
               "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
               "uuid,power.draw,power.limit",
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
                     "tight": (total - used) < TIGHT_MIB,
                     "util": int(p[4] or 0),
                     "uuid": p[5] if len(p) > 5 and p[5] else None,
                     "watts": _float_or_none(p[6]) if len(p) > 6 else None,
                     "watts_limit": _float_or_none(p[7]) if len(p) > 7 else None})
    # The card the main model runs on, by UUID (config.yaml pins `bonsai` by
    # UUID; mcp/power.py uses the same constant). Index order is PCI order and
    # would silently name the wrong card if the cards ever changed slots.
    main_uuid = os.environ.get("YAMADORI_MAIN_GPU_UUID",
                               "GPU-de660e90-0e9c-d465-b389-6df63021b920")
    for r in rows:
        r["main"] = r.get("uuid") == main_uuid
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
                    "started": str(p.get("CreationDate") or "")[:19]})
    return sorted(out, key=lambda x: str(x["what"]))


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


def endpoints() -> list[dict]:
    out = []
    for url, name in (("http://127.0.0.1:11434/v1/models", "llama-swap"),
                      ("http://127.0.0.1:1237/health", "laya")):
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                ok, code = r.status == 200, r.status
        except Exception:                                        # noqa: BLE001
            ok, code = False, 0
        out.append({"name": name, "ok": ok, "code": code,
                    "ms": round((time.time() - t0) * 1000)})
    return out


def context_pool() -> dict:
    try:
        import budget
        return budget.budgets(budget.pool_size(refresh=True))
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


def slots(url: str | None = None) -> dict:
    """Per-slot state from llama-server: idle / prefill / decode, context
    held, and decode and prefill tokens per second (from the change in
    n_decoded since the last read). {"ok": False, "error": ...} when the
    server does not answer within SLOTS_TIMEOUT -- a model swapped out by
    llama-swap is not listening, which is a state, not an outage."""
    t0 = time.time()
    try:
        with urllib.request.urlopen(url or SLOTS_URL, timeout=SLOTS_TIMEOUT) as r:
            data = json.load(r)
    except Exception as e:                                       # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200],
                "slots": [], "ms": round((time.time() - t0) * 1000)}
    if not isinstance(data, list):
        return {"ok": False, "error": "unexpected /slots shape", "slots": [],
                "ms": round((time.time() - t0) * 1000)}
    now = time.time()
    with _lock:
        rows = [_slot_row(s, now) for s in data if isinstance(s, dict)]
    return {"ok": True, "slots": rows, "ms": round((now - t0) * 1000),
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
    try:
        import budget
        ctx = budget.budgets()        # cached pool: no request unless unset
    except Exception as e:                                       # noqa: BLE001
        ctx = {"error": f"{type(e).__name__}: {e}"}
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
                            f"free (under {TIGHT_MIB}); a large prefill spike may not fit")
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
