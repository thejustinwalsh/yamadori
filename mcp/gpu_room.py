#!/usr/bin/env python
"""The A4000's room: what may load onto the card, and what leaves to make space.

THE RULE (operator, 2026-09-24, verbatim)

    "Everything on the A4000 needs to co-operate, swap, and be coordinated.
    If it fits with headroom fine, if it doesn't drop them and load in what
    you need on use."

WHY THIS EXISTS (docs/SELF-IMPROVEMENT-LOG.md #16)

The A4000 (16 GB) holds the search models (embeddings + reranker, llama-swap
group `retrieval`), Laya (not managed by llama-swap, always resident), and two
on-demand consumers: the image generator (`imagegen` / `imagegen-turbo`) and
the vision copy of the 27B (`bonsai-vision`). llama-swap's group flags act
only when an EXCLUSIVE member loads, so "look at an image, then draw one, then
search" ended with everything resident: the live gate of 2026-09-24 read
16,068 of 16,376 MiB, 308 MiB free. The operator's free-VRAM target is ~1.3 GB.

WHAT IT DOES

Every code path about to send the request that makes llama-swap load an A4000
model wraps that request in `use(model_id, upstream=...)`:

  1. llama-swap's `GET /running` (which never loads anything) says what is
     loaded; nvidia-smi, by the card's UUID, says what is free. When the one
     a decision needs cannot be read, an on-demand model (vision, the image
     servers) is REFUSED with A4000_UNREADABLE -- never a load into an
     unknown card; the resident search models (RESIDENT) go ahead.
  2. SIZES says what the model needs: its measured peak where one exists, an
     estimate labelled as such where not. Each row names its source.
  3. Already loaded (and nothing more to allocate), or free - need >=
     HEADROOM_MIB: nothing happens.
  4. Otherwise it unloads other A4000 models through llama-swap's
     `POST /api/models/unload/<id>` -- the same endpoint scripts/watchdog.ps1
     uses -- ONE AT A TIME, least recently used first, re-reading nvidia-smi
     after each, and stops as soon as the model fits. A model another request
     is using right now (a lease) is never unloaded; the caller waits for it.
  5. If it cannot fit even with everything it may unload gone, it unloads
     NOTHING and raises NoRoom: the situation, whether retrying helps (as a
     fact), and a remedy with an owner (AGENTS.md "Failure returns carry the
     next step"). It never lets a load go into an out-of-memory.
  6. Decisions are serialised ACROSS PROCESSES by a file lock: the proxy, the
     tools API (code search -> embeddings) and the worker (hint indexing ->
     embeddings) all load A4000 models, and a lock inside one process would
     not stop the other two. A caller that may allocate (a load, or a draw
     whose server grows from idle to peak) holds the card's room lock until
     its request has finished, so a second caller measures the card AFTER
     the first one's memory is on it -- never two loads racing into the same
     free megabytes. A caller whose model is loaded and allocates nothing
     more takes only a lease (it pins the model) and never waits.
  7. Every decision goes to the process log and, where a request exists, to
     `x_yamadori.gpu_room` (the proxy sets `recording()` per request).

WHAT IT NEVER DOES

  - touch the main model (`bonsai` on the 5060 Ti) or anything not in SIZES:
    `unload()` refuses them itself, whatever the caller asks;
  - count Laya as evictable: it is not llama-swap's, cannot be unloaded, and
    its share is already inside the measured free figure;
  - load anything: it only makes room, and the caller's own request loads.

llama-swap's groups stay as the BACKSTOP (config.yaml `groups:`). `ondemand`
is still exclusive, so a vision load still evicts retrieval and imagegen even
if the coordinator left them -- which the numbers require anyway (search +
vision does not fit with headroom; search + image generator does, ~2.2 GB
free). The coordinator owns the decision; the groups only enforce the same
outcome if something bypasses it.

KNOWN GAP. The proxy's chat path for the unadvertised name `yamadori-vision`
(a client asking for the vision copy directly) calls `ensure_room()` once
before the turn and holds no lock through the generation, so a concurrent
load can still race it. The tool path (describe_image) holds the lock.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.abspath(os.path.join(HERE, "..", "index"))

# The card, by UUID (config.yaml pins every A4000 model with
# CUDA_VISIBLE_DEVICES=<this>; `nvidia-smi -L` lists it).
CARD_UUID = os.environ.get("YAMADORI_A4000_UUID",
                           "GPU-43e37d0c-4104-9056-2552-6109d4d3382c")

# THE OPERATOR'S FREE-VRAM TARGET, "~1.3 GB" (docs/SELF-IMPROVEMENT-LOG.md
# #16; config.yaml's `bonsai` comment uses the same figure for the 5060 Ti).
# Read as 1.3 GiB = 1,331 MiB, the stricter of the two readings. A choice,
# not a measurement: nothing here measured how little free memory the A4000's
# consumers survive. vitals.TIGHT_MIB (1,280) is the dashboard's red line.
HEADROOM_MIB = int(os.environ.get("YAMADORI_A4000_HEADROOM_MIB", "1331"))

# How long a caller waits for the card's room lock or for a model another
# request is using to be released. 300 s covers the longest measured draw
# (1344x1344, 220.5 s, docs/IMAGEGEN.md). A choice.
ROOM_WAIT_S = float(os.environ.get("YAMADORI_GPU_ROOM_WAIT", "300"))
# The shared state file's lock is held for milliseconds; this is a breaker.
STATE_WAIT_S = 30.0
# After an unload, how long to wait for the process to exit and its memory to
# show as free. Not measured; llama-swap's unload stops the process first.
SETTLE_S = float(os.environ.get("YAMADORI_GPU_ROOM_SETTLE", "15"))
# While a model it must unload is in use, re-check this often (each check is
# one /running and one nvidia-smi read).
WAIT_POLL_S = 0.5
# A lease older than this is stale whatever its process says (the longest
# request timeout in the stack is model.TIMEOUT, 3600 s).
LEASE_MAX_S = 3600.0

# The main model and its alias. On the other card; never touched. `unload()`
# also refuses anything not in SIZES, so this is belt and braces.
NEVER = frozenset({"bonsai", "bonsai-agent"})

# The search models: RESIDENT by configuration (llama-swap group `retrieval`,
# loaded at start and kept). When llama-swap's /running cannot be read, a
# request to one of these goes ahead (recorded `uncoordinated`): it is
# presumed loaded, and a request to a loaded one allocates nothing. Every
# other A4000 model is ON DEMAND -- a request to it is a load -- and is
# REFUSED when the card cannot be read (operator rule: never load into an
# out-of-memory; pre-deploy review, 2026-09-24). Until then an unreadable
# /running or nvidia-smi let any load through as `uncoordinated`.
RESIDENT = frozenset({"embeddings", "reranker"})


@dataclass(frozen=True)
class Size:
    """What one A4000 model holds, in MiB.

    peak_mib      the most it holds while serving a request, counted from not
                  loaded (what a load into an empty slot of the card needs)
    resident_mib  what it holds loaded and idle -- what unloading it frees
    measured      True only when peak_mib is a reading of THIS model
    source        where the numbers come from
    """
    peak_mib: int
    resident_mib: int
    measured: bool
    source: str

    @property
    def grows(self) -> bool:
        """It allocates more per request than it holds idle (sd-server
        stages its weights in per generation under --max-vram)."""
        return self.peak_mib > self.resident_mib


# THE SIZE TABLE. Every A4000 model in config.yaml has a row (test_gpu_room
# checks that against the config's CUDA_VISIBLE_DEVICES pins), and nothing
# else does. llama-server allocates weights, KV and compute buffers at load,
# so for those rows loaded == peak.
SIZES: dict[str, Size] = {
    "embeddings": Size(
        2100, 2100, False,
        "ESTIMATE, not measured alone: weights 610 MiB "
        "(Qwen3-Embedding-0.6B-Q8_0.gguf, 639,150,592 B) + f16 KV at -c 8192 "
        "(28 layers x 8 KV heads x 128 x 2 x 2 B = 112 KiB/token, 896 MiB) + "
        "compute buffer and CUDA context (~600, assumed). Measured only "
        "together: embeddings + reranker + Laya = 7,565 MiB (config.yaml "
        "`retrieval` comment; 7,567 in docs/CLM-EVAL.md s.4, 2026-09-24)"),
    "reranker": Size(
        3000, 3000, False,
        "ESTIMATE, not measured alone: weights 610 MiB "
        "(Qwen3-Reranker-0.6B-Q8_0.gguf) + f16 KV at -c 16384 (1,792 MiB) + "
        "compute buffer and CUDA context (~600, assumed). Measured only "
        "together with embeddings and Laya: 7,565 MiB (config.yaml)"),
    "bonsai-vision": Size(
        9449, 9449, False,
        "ESTIMATE, the upper end of config.yaml's 8,265-9,449 MiB (weights "
        "5.95 GiB + mmproj 0.59 + q8 KV at -c 16384 + a 1-2 GiB compute "
        "buffer; `ondemand` group comment, docs/IMAGEGEN.md 'VRAM on the "
        "A4000'). Never measured alone"),
    "imagegen": Size(
        6389, 319, True,
        "MEASURED: peak +6,389 MiB at 1344x1344 under --max-vram 6 (n=1); "
        "+6,281-6,371 at 1024x1024 (n=13); +319 MiB held idle after a "
        "generation (n=1). docs/IMAGEGEN.md 'Also measured'"),
    "imagegen-turbo": Size(
        6389, 319, False,
        "The same sd-server, encoder, VAE and --max-vram 6 budget as "
        "`imagegen`, whose measured ceiling (+6,389) is used. Its own peak was "
        "read once, +5,527 MiB (sd-cli smoke test, n=1, docs/IMAGEGEN.md "
        "'Smoke test'); its idle hold is assumed equal to imagegen's 319"),
    "critic-disabled": Size(
        20173, 20173, False,
        "ARITHMETIC in config.yaml: ~19.7 GiB (weights 15.41 + q8 KV 1.06 + "
        "compute ~2.0 + MTP ~1.2). Does not fit the card"),
}

# llama-swap `swap: true` groups on the A4000 (config.yaml `imagegen`): a load
# of one member stops the other anyway, so the other is unloaded first.
SWAP_GROUPS = (frozenset({"imagegen", "imagegen-turbo"}),)

# Fixed residents that are not llama-swap's. Informational: their memory is
# inside the measured free figure, and nothing here can unload them.
FIXED = {
    "laya": ("always resident (mcp/laya_service.py, port 1237); not measured "
             "separately -- Windows reports per-process GPU memory as N/A "
             "(docs/CLM-EVAL.md s.4). '~1 GiB' per the laya_service comment; "
             ">= 2,445 MiB by docs/IMAGEGEN.md's arithmetic"),
}

PROCESS = os.path.basename(sys.argv[0] or "python") or "python"


def enabled() -> bool:
    """On unless YAMADORI_GPU_ROOM is 0/off. scripts/run_tests.py turns it off
    for the offline suites so no test can reach the real llama-swap."""
    return os.environ.get("YAMADORI_GPU_ROOM", "1").strip().lower() not in (
        "0", "off", "false", "no")


def on_card(model: str | None) -> bool:
    return bool(model) and model in SIZES


def default_upstream() -> str:
    return os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")


# ------------------------------------------------------------------- readers
def running(upstream: str) -> list[dict] | None:
    """llama-swap's `GET /running` rows, or None if it cannot be read. This
    endpoint reports; it never loads anything."""
    try:
        with urllib.request.urlopen(f"{upstream.rstrip('/')}/running",
                                    timeout=5) as r:
            d = json.loads(r.read().decode("utf-8") or "{}")
    except Exception:                                            # noqa: BLE001
        return None
    rows = d.get("running") if isinstance(d, dict) else None
    if not isinstance(rows, list):
        return None
    return [x for x in rows if isinstance(x, dict) and x.get("model")]


def _nvidia_smi() -> dict | None:
    import vitals
    for r in vitals.gpus(timeout=10):
        if (r.get("uuid") or "").lower() == CARD_UUID.lower():
            return {"free_mib": r["free_mib"], "used_mib": r["used_mib"],
                    "total_mib": r["total_mib"]}
    return None


# Tests replace this with a fake card.
CARD_READER = _nvidia_smi


def card() -> dict | None:
    """{free_mib, used_mib, total_mib} of the A4000, or None."""
    try:
        c = CARD_READER()
    except Exception:                                            # noqa: BLE001
        return None
    return c if isinstance(c, dict) and "free_mib" in c else None


def unload(upstream: str, model: str) -> str | None:
    """Ask llama-swap to stop `model`. None on success, else why not. Refuses
    anything that is not an A4000 on-demand model, whoever asks."""
    if model in NEVER or model not in SIZES:
        return f"refused: {model!r} is not an A4000 model this coordinator may unload"
    req = urllib.request.Request(
        f"{upstream.rstrip('/')}/api/models/unload/{urllib.parse.quote(model, safe='')}",
        data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
        return None
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:                                       # noqa: BLE001
        return f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------- locks
def _lock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, 0, 0)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, 0, 0)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


class _FileLock:
    """A lock held across processes (an OS file lock) and threads (an RLock
    in front of it). Re-entrant for the thread that holds it. The OS drops a
    file lock when its process dies, so a crash cannot leave the card locked."""

    def __init__(self, path: str):
        self.path = path
        self._t = threading.RLock()
        self._depth = 0
        self._fd: int | None = None

    def acquire(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(timeout, 0.0)
        if not self._t.acquire(timeout=max(timeout, 0.001)):
            return False
        if self._depth == 0:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            while True:
                try:
                    _lock_fd(fd)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        os.close(fd)
                        self._t.release()
                        return False
                    time.sleep(0.05)
            self._fd = fd
        self._depth += 1
        return True

    def release(self) -> None:
        self._depth -= 1
        if self._depth == 0 and self._fd is not None:
            try:
                _unlock_fd(self._fd)
            finally:
                os.close(self._fd)
                self._fd = None
        self._t.release()


_LOCKS: dict[str, _FileLock] = {}
_LOCKS_GUARD = threading.Lock()


def state_dir() -> str:
    return os.environ.get("YAMADORI_GPU_ROOM_DIR") or INDEX


def _lock(name: str) -> _FileLock:
    path = os.path.join(state_dir(), f"gpu_room.{name}.lock")
    with _LOCKS_GUARD:
        if path not in _LOCKS:
            _LOCKS[path] = _FileLock(path)
        return _LOCKS[path]


def room_lock() -> _FileLock:
    """Held by a caller that may allocate, until its request has finished."""
    return _lock("room")


# --------------------------------------------------------------------- state
# {"last_use": {model: epoch}, "leases": {id: {model, pid, t, until}},
#  "evicting": {model: {pid, t}}} -- shared by every process, read and
# written only under the state lock.
def _state_path() -> str:
    return os.path.join(state_dir(), "gpu_room.json")


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:                                            # noqa: BLE001
        return True


def _load() -> dict:
    try:
        with open(_state_path(), encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}
    if not isinstance(st, dict):
        st = {}
    for k in ("last_use", "leases", "evicting"):
        if not isinstance(st.get(k), dict):
            st[k] = {}
    now = time.time()
    st["leases"] = {i: v for i, v in st["leases"].items()
                    if isinstance(v, dict) and v.get("until", 0) > now
                    and _alive(int(v.get("pid", 0)))}
    st["evicting"] = {m: v for m, v in st["evicting"].items()
                      if isinstance(v, dict) and now - v.get("t", 0) < 120
                      and _alive(int(v.get("pid", 0)))}
    return st


def _save(st: dict) -> None:
    """Write under the state lock. On Windows a replace can be refused for a
    moment by something outside the stack holding the file (a scanner); a
    few retries, then the write is dropped and said so -- a lost lease or
    last-use time costs one decision's accuracy, not the request."""
    path = _state_path()
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    for i in range(5):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(st, f)
            os.replace(tmp, path)
            return
        except OSError as e:
            if i == 4:
                print(f"  gpu_room[{PROCESS}]: state not saved ({type(e).__name__}: "
                      f"{e})", flush=True)
                return
            time.sleep(0.05)


@contextlib.contextmanager
def _state():
    lk = _lock("state")
    if not lk.acquire(STATE_WAIT_S):
        raise NoRoom("A4000_BUSY",
                     f"The A4000 coordinator's state lock was not free within "
                     f"{STATE_WAIT_S:.0f}s. Nothing was loaded.",
                     True, [_agent_retry()], {"why": "state lock timeout"})
    try:
        st = _load()
        yield st
        _save(st)
    finally:
        lk.release()


_lease_n = 0
_lease_guard = threading.Lock()


def _add_lease(st: dict, model: str) -> str:
    global _lease_n
    with _lease_guard:
        _lease_n += 1
        lid = f"{os.getpid()}-{threading.get_ident()}-{_lease_n}"
    now = time.time()
    st["leases"][lid] = {"model": model, "pid": os.getpid(), "t": now,
                         "until": now + LEASE_MAX_S}
    st["last_use"][model] = now
    return lid


# ------------------------------------------------------------------ failures
class NoRoom(RuntimeError):
    """The model cannot be loaded onto the A4000 now. Carries the situation,
    whether retrying can help (as a fact), remedies with an owner, and the
    decision record."""

    def __init__(self, code: str, reason: str, retryable: bool,
                 remedies: list[dict], decision: dict):
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.retryable = retryable
        self.remedies = remedies
        self.decision = decision

    @property
    def facts(self) -> dict:
        d = self.decision or {}
        return {k: d[k] for k in ("model", "need_mib", "free_mib",
                                  "headroom_mib", "evicted", "in_use")
                if k in d}

    def envelope(self, tool: str) -> dict:
        return {"tool": tool, "ok": False, "error": self.code,
                "reason": self.reason, "retryable": self.retryable,
                "remedies": self.remedies, **self.facts}


def _agent_retry() -> dict:
    return {"fixable_by": "agent",
            "action": "call the same tool again in a minute",
            "effect": ("the model holding the A4000 is released when its "
                       "request finishes, and the next call makes room")}


def _no_room_remedies(model: str) -> list[dict]:
    return [{"fixable_by": "operator",
             "action": (f"read nvidia-smi for the A4000 ({CARD_UUID}): memory "
                        f"not held by llama-swap's A4000 models is Laya's or "
                        f"a process outside the stack; stop the outsider, or "
                        f"lower YAMADORI_A4000_HEADROOM_MIB (now "
                        f"{HEADROOM_MIB})"),
             "why_not_the_agent": "the agent cannot see or free the GPU"},
            {"fixable_by": "agent",
             "action": f"tell the user `{model}` cannot run on the server right now",
             "effect": "retrying in this turn cannot change the card's contents"}]


# ----------------------------------------------------------------- recording
_tl = threading.local()
_logged: dict[str, float] = {}
_log_guard = threading.Lock()
# A fast-path "loaded" line per model at most this often (the worker embeds
# thousands of batches); every other decision is logged every time.
LOG_LOADED_EVERY_S = 60.0


@contextlib.contextmanager
def recording(sink: list | None):
    """Decisions made in this thread go to `sink` (x_yamadori.gpu_room)."""
    prev = getattr(_tl, "sink", None)
    _tl.sink = sink if sink is not None else prev
    try:
        yield
    finally:
        _tl.sink = prev


def _log(d: dict) -> None:
    if d.get("action") == "loaded":
        with _log_guard:
            now = time.time()
            if now - _logged.get(d["model"], 0.0) < LOG_LOADED_EVERY_S:
                return
            _logged[d["model"]] = now
    ev = ",".join(f"{e['model']}(+{e.get('freed_mib')})" + (
        f"!{e['error']}" if e.get("error") else "")
        for e in d.get("evicted") or [])
    free = (f" free={d['free_before_mib']}->{d.get('free_after_mib')}"
            if d.get("free_before_mib") is not None else "")
    print(f"  gpu_room[{PROCESS}]: {d.get('model')} {d.get('action')}"
          f" need={d.get('need_mib')}{free} headroom={d.get('headroom_mib')}"
          + (f" evicted=[{ev}]" if ev else "")
          + (f" waited={d['waited_s']}s" if d.get("waited_s") else "")
          + (f" -- {d['why']}" if d.get("why") else ""), flush=True)


def _emit(d: dict | None) -> None:
    if not d:
        return
    sink = getattr(_tl, "sink", None)
    if sink is not None:
        sink.append(d)
    try:
        _log(d)
    except Exception:                                            # noqa: BLE001
        pass


# ---------------------------------------------------------------- the choice
def _mates(model: str) -> frozenset:
    for g in SWAP_GROUPS:
        if model in g:
            return g - {model}
    return frozenset()


def _order(cands: list[str], model: str, last_use: dict) -> list[str]:
    """Eviction order: the requested model's swap-group mate first (llama-swap
    stops it on this load anyway), then least recently used; a model never
    used through the coordinator (preloaded) counts as oldest; ties go to the
    bigger one, so fewer models leave."""
    mates = _mates(model)
    return sorted(cands, key=lambda m: (0 if m in mates else 1,
                                        float(last_use.get(m, 0.0)),
                                        -SIZES[m].resident_mib, m))


def _settle(upstream: str, victim: str, before: int) -> int | None:
    """Wait for the unloaded model to leave /running and its memory to show
    as free. Returns the MiB freed, as measured."""
    deadline = time.monotonic() + SETTLE_S
    while True:
        run = running(upstream)
        gone = run is not None and victim not in {r["model"] for r in run}
        c = card()
        if gone and c and c["free_mib"] > before:
            return c["free_mib"] - before
        if time.monotonic() >= deadline:
            return (c["free_mib"] - before) if c else None
        time.sleep(0.25)


# ------------------------------------------------------- the chat path waits
# for nothing (pre-deploy review, 2026-09-24). A chat request's own embedding
# (skill / hint selection, in proxy.prepare) went through use() like any
# caller, and when the search model was not loaded it waited up to
# ROOM_WAIT_S (300 s) for the room lock -- which an image draw holds for its
# whole run. Inside `fail_fast()` a caller never waits: a loaded model still
# takes its lease (no room lock), and one that would need the room lock gets
# A4000_BUSY at once, which the caller records and goes on without.
@contextlib.contextmanager
def fail_fast(why: str = "a chat turn does not wait for the A4000"):
    prev = getattr(_tl, "fail_fast", None)
    _tl.fail_fast = why
    try:
        yield
    finally:
        _tl.fail_fast = prev


def _wait_budget() -> tuple[float, str | None]:
    """(seconds this thread may wait for the room, the fail-fast reason)."""
    why = getattr(_tl, "fail_fast", None)
    return (0.0, why) if why else (ROOM_WAIT_S, None)


def _unreadable_remedies(what: str) -> list[dict]:
    return [_agent_retry(),
            {"fixable_by": "operator",
             "action": (f"check that {what} answers (llama-swap on :11434; "
                        f"nvidia-smi for the A4000, {CARD_UUID})"),
             "why_not_the_agent": "the agent cannot see or restart the GPU "
                                  "services"}]


class use:
    """`with gpu_room.use(model_id, upstream=...) as decision:` around the
    request that would make llama-swap load `model_id`. Raises NoRoom (or
    `on_no_room(err)` when given) when the model cannot fit; `decision` is
    None for a model not on the A4000 or with the coordinator off."""

    def __init__(self, model: str | None, upstream: str | None = None, *,
                 on_no_room=None):
        self.model = model
        self.upstream = upstream or default_upstream()
        self.on_no_room = on_no_room
        self.decision: dict | None = None
        self._lease: str | None = None
        self._room = False
        self._t0 = time.time()

    def __enter__(self) -> dict | None:
        try:
            self.decision = self._enter()
        except NoRoom as e:
            _emit(e.decision)
            if self.on_no_room is not None:
                raise self.on_no_room(e) from None
            raise
        _emit(self.decision)
        return self.decision

    def __exit__(self, *exc) -> bool:
        try:
            if self._lease:
                try:
                    with _state() as st:
                        st["leases"].pop(self._lease, None)
                        st["last_use"][self.model] = time.time()
                except Exception:                                # noqa: BLE001
                    pass            # the lease expires on its own
                self._lease = None
            if self._room and self.decision is not None:
                c = card()
                self.decision["free_end_mib"] = c["free_mib"] if c else None
        finally:
            if self._room:
                self._room = False
                room_lock().release()
        return False

    # ------------------------------------------------------------------
    def _base(self) -> dict:
        s = SIZES[self.model]
        return {"model": self.model, "card": "A4000",
                "headroom_mib": HEADROOM_MIB, "measured": s.measured,
                "process": PROCESS, "pid": os.getpid()}

    def _done(self, d: dict, **kw) -> dict:
        d.update(kw)
        d["ms"] = round((time.time() - self._t0) * 1000)
        return d

    def _enter(self) -> dict | None:
        if not on_card(self.model) or not enabled():
            return None
        size = SIZES[self.model]
        d = self._base()
        run = running(self.upstream)
        if run is None:
            if self.model in RESIDENT:
                return self._done(d, action="uncoordinated", why=(
                    f"llama-swap /running at {self.upstream} did not answer; "
                    f"`{self.model}` is resident by configuration, so the "
                    f"request goes ahead (it allocates nothing when loaded)"))
            raise NoRoom(
                "A4000_UNREADABLE",
                f"llama-swap's /running at {self.upstream} did not answer, so "
                f"what is on the A4000 is unknown and `{self.model}` "
                f"(on demand, {size.peak_mib:,} MiB) was NOT loaded: a load "
                f"the card cannot be read for could go into an out-of-memory. "
                f"Nothing was loaded or unloaded.",
                True, _unreadable_remedies("llama-swap /running"),
                self._done(d, action="unreadable", need_mib=size.peak_mib,
                           why="/running did not answer"))
        names = {r["model"] for r in run}
        # FAST PATH: loaded, and a request allocates nothing more. A lease
        # pins it; no nvidia-smi read, no room lock.
        if self.model in names and not size.grows:
            with _state() as st:
                if self.model not in st["evicting"]:
                    self._lease = _add_lease(st, self.model)
            if self._lease:
                return self._done(d, action="loaded", need_mib=0, loaded=True)
        # SLOW PATH: this request may allocate. Hold the card's room until
        # it has finished.
        tw = time.time()
        budget, fast = _wait_budget()
        self._budget = budget
        if not room_lock().acquire(budget):
            d.update(need_mib=size.peak_mib, waited_s=round(time.time() - tw, 1))
            if fast:
                raise NoRoom(
                    "A4000_BUSY",
                    f"`{self.model}` is not loaded, and making room for it "
                    f"needs the A4000, which another load holds (an image "
                    f"draw holds it for its whole run). {fast}, so it did "
                    f"not wait; nothing was loaded.",
                    True, [_agent_retry()],
                    self._done(d, action="busy", fail_fast=True,
                               why="fail-fast: the room lock was held"))
            raise NoRoom(
                "A4000_BUSY",
                f"Another load onto the A4000 held the card for "
                f"{budget:.0f}s (a generation in progress), so "
                f"`{self.model}` was not loaded.",
                True, [_agent_retry()],
                self._done(d, action="busy", why="room lock timeout"))
        self._room = True
        try:
            return self._make_room(d, size, tw)
        except BaseException:
            self._room = False
            room_lock().release()
            raise

    def _make_room(self, d: dict, size: Size, tw: float) -> dict:
        model, up = self.model, self.upstream
        deadline = tw + getattr(self, "_budget", ROOM_WAIT_S)
        evicted: list[dict] = []
        failed: set[str] = set()
        first_free = None
        waited_for: set[str] = set()
        while True:
            if len(evicted) > len(SIZES):
                # A breaker, never expected: every eviction removes a model
                # or marks it failed, so this cannot repeat past the table.
                raise NoRoom("A4000_NO_ROOM",
                             f"The coordinator unloaded {len(evicted)} models "
                             f"for `{model}` and stopped (a defect). Nothing "
                             f"was loaded.", False, _no_room_remedies(model),
                             self._done(d, action="no_room", evicted=evicted,
                                        why="eviction breaker"))
            run = running(up)
            if run is None:
                self._unreadable(d, size, f"llama-swap /running at {up} "
                                          f"stopped answering", evicted)
            names = [r["model"] for r in run]
            c = card()
            if c is None:
                self._unreadable(d, size, f"nvidia-smi did not report the "
                                          f"A4000 ({CARD_UUID})", evicted)
            loaded = model in names
            need = (size.peak_mib - size.resident_mib) if loaded else size.peak_mib
            free = c["free_mib"]
            if first_free is None:
                first_free = free
            common = dict(need_mib=need, loaded=loaded,
                          free_before_mib=first_free, free_after_mib=free,
                          free_mib=free, evicted=evicted,
                          waited_s=round(time.time() - tw, 1),
                          in_use=sorted(waited_for))
            if free - need >= HEADROOM_MIB:
                with _state() as st:
                    self._lease = _add_lease(st, model)
                action = ("evicted" if evicted else
                          "loaded" if loaded and need == 0 else "fit")
                return self._done(d, action=action, **common)
            wait = False
            with _state() as st:
                pinned = {v["model"] for v in st["leases"].values()}
                on = [m for m in names if on_card(m) and m != model
                      and m not in NEVER and m not in failed]
                free_c = [m for m in on if m not in pinned
                          and m not in st["evicting"]]
                busy = [m for m in on if m not in free_c]
                est = free + sum(SIZES[m].resident_mib for m in free_c)
                if est - need < HEADROOM_MIB:
                    est_all = est + sum(SIZES[m].resident_mib for m in busy)
                    helps = bool(busy) and est_all - need >= HEADROOM_MIB
                    if helps and time.time() < deadline:
                        waited_for |= set(busy)
                        wait = True
                    else:
                        self._no_room(d, common, free_c, busy, est, helps)
                else:
                    victim = _order(free_c, model, st["last_use"])[0]
                    st["evicting"][victim] = {"pid": os.getpid(),
                                              "t": time.time()}
            if wait:
                time.sleep(WAIT_POLL_S)
                continue
            err = unload(up, victim)
            freed = _settle(up, victim, free) if err is None else None
            with _state() as st:
                st["evicting"].pop(victim, None)
            evicted.append({"model": victim, "freed_mib": freed,
                            **({"error": err} if err else {})})
            if err is not None:
                failed.add(victim)

    def _unreadable(self, d: dict, size: Size, why: str,
                    evicted: list) -> None:
        """The slow path could not read the card: this request would
        allocate (the model is not loaded, or it grows per request), so it
        is refused -- never a load into an unknown card."""
        raise NoRoom(
            "A4000_UNREADABLE",
            f"{why}, so the A4000's free memory is unknown and `{self.model}` "
            f"(which would allocate up to {size.peak_mib:,} MiB) was NOT "
            f"loaded: a load the card cannot be read for could go into an "
            f"out-of-memory. Nothing was loaded"
            + (f"; {', '.join(e['model'] for e in evicted)} had already been "
               f"unloaded" if evicted else "") + ".",
            True, _unreadable_remedies("llama-swap /running and nvidia-smi"),
            self._done(d, action="unreadable", need_mib=size.peak_mib,
                       evicted=evicted, why=why))

    def _no_room(self, d: dict, common: dict, free_c: list, busy: list,
                 est: int, helps: bool) -> None:
        model, need, free = self.model, common["need_mib"], common["free_mib"]
        rec = self._done(d, action="busy" if helps else "no_room",
                         candidates=free_c, **common)
        if helps:
            raise NoRoom(
                "A4000_BUSY",
                f"`{model}` needs {need:,} MiB on the A4000 plus {HEADROOM_MIB:,} "
                f"headroom and {free:,} is free; making room means unloading "
                f"{', '.join(busy)}, which another request is using. Waited "
                f"{rec['waited_s']:.0f}s. Nothing was loaded.",
                True, [_agent_retry()], rec)
        done = [e["model"] for e in common["evicted"]]
        raise NoRoom(
            "A4000_NO_ROOM",
            f"The A4000 cannot hold `{model}`: it needs {need:,} MiB "
            f"({'measured' if SIZES[model].measured else 'estimated'}) plus "
            f"{HEADROOM_MIB:,} MiB headroom, {free:,} MiB is free, and "
            f"unloading every model the coordinator may touch "
            f"({', '.join(free_c) or 'none is loaded'}) would bring it to "
            f"about {est:,}. Nothing was loaded"
            + (f"; {', '.join(done)} had already been unloaded because the "
               f"size table was optimistic" if done else
               " and nothing was unloaded") + ".",
            False, _no_room_remedies(model), rec)


def ensure_room(model: str | None, upstream: str | None = None) -> dict | None:
    """One-shot: make room for `model` and return the decision, holding no
    lock afterwards. Raises NoRoom. For a caller that cannot wrap its request
    (see KNOWN GAP in the module docstring); everything else uses `use()`."""
    u = use(model, upstream)
    dec = u.__enter__()
    u.__exit__(None, None, None)
    return dec


def describe() -> dict:
    """The table and settings, for the dashboard or a person reading."""
    return {"card_uuid": CARD_UUID, "headroom_mib": HEADROOM_MIB,
            "enabled": enabled(), "never": sorted(NEVER),
            "sizes": {m: {"peak_mib": s.peak_mib,
                          "resident_mib": s.resident_mib,
                          "measured": s.measured, "source": s.source}
                      for m, s in SIZES.items()},
            "fixed": FIXED}


if __name__ == "__main__":
    print(json.dumps(describe(), indent=1))
