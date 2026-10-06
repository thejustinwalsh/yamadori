#!/usr/bin/env python
"""A pre-read of Flash-Next's expert file into the OS file cache (AGENTS.md "Flash-Next's first prompt").

WHAT AND WHY. Flash-Next's experts (IQ2_XS GGUF shard 1, 39.2 GB, mmapped, unpinned: LLAMA_PIN_EXPERTS=0 in config.yaml)
are demand-paged from disk during the FIRST 8,192-token batch of the first prompt, at ~0.4 GB/s: 31.5-31.9 GB read,
batch 1 75-87 s against 16.2-16.9 s warm, ttft 201-211 s for a 24.6K prompt from a cold file cache
(bench/fn_first_prompt.py, bench/results/fn_first_prompt/20261006-a, n=3 per arm). A sequential read of shard 1 with
16 MiB cached reads BEFORE the load took 17.0-17.4 s (2.3 GB/s), left 42.5-44.3 GB available (evictable cache), and cut
the prefill's disk reads to 1.6-3.8 GB and ttft to 83-103 s. An idle loaded server (~4 h, n=1) had its working set
trimmed to 67 MB and its next prompt read 24.6 GB from disk (ttft 99.6 s): the penalty follows a long idle, too.

THE FILE is the `-m` argument of the model's llama-swap entry (config.yaml, macros expanded: catalog.swap_config()), never
a path written here. ONLY THE FIRST SHARD: it holds the attention, router and experts; shard 2 (26.8 GB, the n-gram table,
read by rows) is not read, because both together (63.4 GB) exceed what the file cache holds and would push shard 1 out
(the measurement's notes.txt). Which models get a pre-read is the tier table's row key `preread` (mcp/tier_models.yaml:
flash-next), not a model name here.

WHEN (mcp/max_mode.py wait_ready calls this; the request's heartbeats keep flowing because the wait runs under
`card_wait`, which proxy._TurnPump reads):
  swap     the request makes llama-swap load the model: the read starts as the swap request goes out and the request
           waits for it before its prompt is sent (OVERLAP, the default: the read and the load share the disk -- whether
           that helps or contends is UNMEASURED), or the read runs first and the load follows (switch `preread_overlap`
           off). The cold load itself took 68-89 s, against 14-24 s after a pre-read (same bench).
  trimmed  the model is loaded and the llama-server process's working set is below HALF the file's size. The experts
           are mapped file pages that must be resident for the prefill not to hard-fault; the measured working sets fall
           in two groups -- trimmed 67 MB (4 h idle, n=1) and ~900 MB, warm 39.1-44.4 GB after a prompt (the warm 39.1 GB
           is just UNDER the 39.23 GB file, so "below the file's size" would re-read on ordinary turns) -- and half the
           file sits inside that gap (decide_trimmed). The proxy cannot know prefix reuse before it sends, so the working
           set alone decides.

ONLY WHEN THE CACHE HAS ROOM FOR THE WHOLE FILE (decide_room; found 2026-10-06, bench/results/fn_first_prompt/
20261006-proxy and -proxy-b, n=3 per arm): a sequential read of a file larger than the memory available to the file cache
evicts its own head, and the prefill reads from the head. With ~48 GB available (free + standby, before the request) the
overlapped read left 1.7-1.8 GB of disk reads in the prefill and ttft 91-99 s, against 165-174 s with the priority-2
read and 163-167 s with no read (n=3, same box); with ~38 GB available (another model's host memory, imagegen-turbo, was
resident) the read-then-load swap cost ttft 191-205 s and the overlapped one 240 s (n=3 / n=1), against the 163-167 s of no
read at all, and the trimmed-server read (33-35 GB available) waited 26 s for nothing (prefill 26-29 GB from disk,
n=3). So the read is skipped, with the numbers recorded, when the memory available is below the file's size. The margin
between 39 GB (the file) and 47 GB (the largest measured working case) is UNMEASURED.

ONE READ AT A TIME (`_read_lock`): a read in flight for the same file is JOINED, never restarted; a read of another file
waits its turn. The read is bounded by max_mode.LOAD_TIMEOUT_S (llama-swap's own 900 s health-check timeout, the longest a
load may already take): after that the request goes on and the record says `timed_out`.

THE READ RUNS AT NORMAL MEMORY AND I/O PRIORITY (found 2026-10-06; the cause is INFERRED, the fix not yet re-measured live: the stack-process priorities and the six swaps are measured, bench/results/fn_first_prompt/20261006-proxy, n=3 per
arm, and a probe of the stack's processes): every process of the stack (the proxy, llama-server) runs at BELOW_NORMAL with
MEMORY priority 2 and I/O priority low, against 5 and normal for the bench's shell. Pages a read puts in the file cache are
tagged with the reader's memory priority, and the standby list gives up its lowest priority first: the first proxy version
(a thread inheriting priority 2) read the file in 26-31 s and the load, which takes ~40 GB of private memory, then
consumed exactly those pages -- the prefill still read 31.1-33.1 GB from disk (6 of 6 swaps, overlap and read-then-load
alike), where the same read from the bench (priority 5) left 1.6-3.8 GB. So the read thread sets ITS OWN memory priority
to 5 (SetThreadInformation ThreadMemoryPriority, MEMORY_PRIORITY_NORMAL) and the file's I/O hint to normal
(SetFileInformationByHandle FileIoPriorityHintInfo, IoPriorityHintNormal): both are allowed from a priority-2 process
(probed) and are recorded in x_yamadori.preread.priority. The request waits for this read, so a low I/O priority could
only make the waiter wait longer. (THREAD_MODE_BACKGROUND_BEGIN would do the opposite: it lowers memory priority too.)

SWITCHES (tiers.BEHAVIOURS, default ON): `preread` (YAMADORI_PREREAD; X-Yamadori-Features {"preread": false}) and
`preread_overlap` (YAMADORI_PREREAD_OVERLAP; {"preread_overlap": false} = read, then load). x_yamadori.preread and one
log line record every decision.

    python mcp/preread.py status [MODEL]      read-only: the file, its size, the server's pid and working set
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The measured arm's read size (bench/fn_first_prompt.py read_files: 16 MiB cached reads, 2.25-2.31 GB/s).
CHUNK = 16 << 20
# llama-swap's own healthCheckTimeout (config.yaml, 900 s; max_mode.LOAD_TIMEOUT_S): the longest a load may take.
TIMEOUT_S = 900
POLL_S = 0.5
FILE_RULE = ("the entry's -m file, the first shard of the split GGUF; shard 2 (the n-gram table) is not read: both "
             "together exceed what the file cache holds and would push shard 1 out (fn_first_prompt 20261006-a)")

_lock = threading.Lock()              # guards _current
_read_lock = threading.Lock()         # one read at a time
_current: dict[str, "Handle"] = {}    # path -> the read in flight
_pid_cache: dict[str, int] = {}


# ------------------------------------------------------------------ switches --
def switches(features=None) -> dict:
    """{on, source, overlap, overlap_source}: tiers.BEHAVIOURS `preread` and `preread_overlap`, the header's forced
    values first (the same rule as the ledger's restore: read before the tier is resolved)."""
    import tiers
    on, src = tiers.behaviour_of_header(features, "preread")
    ov, ov_src = tiers.behaviour_of_header(features, "preread_overlap")
    return {"on": on, "source": src, "overlap": ov, "overlap_source": ov_src}


# ---------------------------------------------------------------------- file --
def _expand(s: str, macros: dict) -> str:
    for _ in range(5):
        n = re.sub(r"\$\{(\w+)\}", lambda m: str(macros.get(m.group(1), m.group(0))), s)
        if n == s:
            break
        s = n
    return s


def model_file(model: str, config_path: str | None = None) -> tuple[str | None, str]:
    """(the `-m` path of `model`'s llama-swap entry with the macros expanded, why not). Read from the config file."""
    try:
        import catalog
        import yaml
        path = config_path or catalog.swap_config()
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:                                           # noqa: BLE001
        return None, f"llama-swap's config could not be read ({type(e).__name__}: {e})"[:200]
    entry = (cfg.get("models") or {}).get(model)
    if not isinstance(entry, dict) or not entry.get("cmd"):
        return None, f"llama-swap's config has no cmd for {model}"
    macros = dict(cfg.get("macros") or {})
    lines = [ln for ln in str(entry["cmd"]).splitlines() if not ln.strip().startswith("#")]
    cmd = _expand(" ".join(lines), macros)
    m = re.search(r"(?:^|\s)(?:-m|--model)\s+(\"[^\"]+\"|'[^']+'|\S+)", cmd)
    if not m:
        return None, f"{model}'s cmd has no -m argument"
    return m.group(1).strip("\"'"), ""


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


# ---------------------------------------------------------------- working set --
def server_pid(path: str) -> int | None:
    """The pid of the llama-server llama-swap runs for this model file: the process whose command line carries the
    file as an argument (llama-swap's /running names no pid). Cached, and checked by its command line each time."""
    import psutil
    want = _norm(path)

    def has(p) -> bool:
        try:
            return any(_norm(a) == want for a in p.cmdline())
        except (psutil.Error, OSError):
            return False
    pid = _pid_cache.get(want)
    if pid is not None:
        try:
            if has(psutil.Process(pid)):
                return pid
        except psutil.Error:
            pass
        _pid_cache.pop(want, None)
    for p in psutil.process_iter(["name"]):
        if "llama-server" in (p.info.get("name") or "").lower() and has(p):
            _pid_cache[want] = p.pid
            return p.pid
    return None


def working_set(path: str) -> tuple[int | None, str]:
    """(the server's working set in bytes, why not). psutil's rss is the Windows WorkingSetSize."""
    try:
        import psutil
        pid = server_pid(path)
        if pid is None:
            return None, "no llama-server process has this file in its command line"
        return int(psutil.Process(pid).memory_info().rss), ""
    except Exception as e:                                           # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"[:200]


# ---------------------------------------------------------------------- read --
_PRIORITY: dict[str, dict] = {}       # path -> what the last read set (read_sequential), for the record


def raise_cache_priority(f) -> dict:
    """THIS thread's memory priority to normal (5) and the open file's I/O hint to normal (2), so the pages the read
    caches are not the first the standby list gives up (see THE READ RUNS AT NORMAL ... above). Windows only; each
    call is best effort and the result says what took: {memory: 5|None, io_hint: 2|None, error?}."""
    out: dict = {"memory": None, "io_hint": None}
    if os.name != "nt":
        return out
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32                                # type: ignore[attr-defined]
        k32.GetCurrentThread.restype = ctypes.c_void_p
        k32.SetThreadInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetFileInformationByHandle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        v = wintypes.ULONG(5)                                        # MEMORY_PRIORITY_NORMAL
        if k32.SetThreadInformation(k32.GetCurrentThread(), 0, ctypes.byref(v), 4):   # ThreadMemoryPriority
            out["memory"] = 5
        h = wintypes.ULONG(2)                                        # IoPriorityHintNormal
        if k32.SetFileInformationByHandle(msvcrt.get_osfhandle(f.fileno()), 12, ctypes.byref(h), 4):  # FileIoPriorityHintInfo
            out["io_hint"] = 2
    except Exception as e:                                           # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:120]
    return out


def read_sequential(path: str, should_stop) -> int:
    """Read the file once, front to back, with ordinary cached reads (the bytes are discarded) at normal memory and I/O
    priority (raise_cache_priority). Stops early when should_stop() says so. Returns the bytes read."""
    n = 0
    buf = bytearray(CHUNK)
    with open(path, "rb", buffering=0) as f:
        _PRIORITY[path] = raise_cache_priority(f)
        while not should_stop():
            k = f.readinto(buf)
            if not k:
                break
            n += k
    return n


_READ = read_sequential               # tests replace it


class Handle:
    """One pre-read, in flight or finished."""

    def __init__(self, path: str, why: str, overlapped: bool, size: int, ws: int | None, model: str):
        self.path, self.why, self.overlapped, self.size, self.ws, self.model = path, why, overlapped, size, ws, model
        self.done = threading.Event()
        self.t_made = time.time()
        self.t_start: float | None = None
        self.t_end: float | None = None
        self.bytes = 0
        self.error = ""
        self.timed_out = False
        self.stop = threading.Event()

    def run(self) -> None:
        try:
            with _read_lock:
                self.t_start = time.time()
                deadline = self.t_start + TIMEOUT_S
                try:
                    self.bytes = int(_READ(self.path, lambda: self.stop.is_set() or time.time() > deadline) or 0)
                except Exception as e:                               # noqa: BLE001
                    self.error = f"{type(e).__name__}: {e}"[:200]
                self.timed_out = bool(not self.error and self.bytes < self.size and time.time() > deadline)
        finally:
            self.t_end = time.time()
            with _lock:
                if _current.get(self.path) is self:
                    _current.pop(self.path, None)
            self.done.set()
            ms = round(((self.t_end - (self.t_start or self.t_made))) * 1000)
            rate = (self.bytes / 1e9) / max(self.t_end - (self.t_start or self.t_made), 1e-6)
            print(f"  preread: {self.model} {os.path.basename(self.path)} {self.bytes / 1e9:.2f} of "
                  f"{self.size / 1e9:.2f} GB in {ms / 1000:.1f} s ({rate:.2f} GB/s), why {self.why}"
                  f"{', overlapped with the load' if self.overlapped else ''}"
                  + (f", working set {self.ws / 1e6:.0f} MB" if self.ws is not None else "")
                  + (f"; ERROR {self.error}" if self.error else "") + ("; TIMED OUT" if self.timed_out else ""),
                  flush=True)

    def wait(self, check=None, timeout: float = TIMEOUT_S, poll: float = POLL_S) -> bool:
        """Block until the read ends (True) or `timeout` passes (False). `check` (cancel.check) is called each poll
        and may raise: the read goes on without this waiter."""
        end = time.time() + timeout
        while not self.done.is_set():
            if check is not None:
                check()
            left = end - time.time()
            if left <= 0:
                return False
            self.done.wait(min(poll, left))
        return True

    def record(self, joined: bool, waited_s: float) -> dict:
        t0 = self.t_start or self.t_made
        t1 = self.t_end or time.time()
        return {"why": self.why, "model": self.model, "file": os.path.basename(self.path), "file_rule": FILE_RULE,
                "file_bytes": self.size, "bytes": self.bytes, "ms": round((t1 - t0) * 1000),
                "gb_per_s": round(self.bytes / 1e9 / max(t1 - t0, 1e-6), 2),
                "overlapped": self.overlapped, "priority": _PRIORITY.get(self.path),
                "working_set_mb": None if self.ws is None else round(self.ws / 1e6),
                "joined": joined, "waited_ms": round(waited_s * 1000), "finished": self.done.is_set(),
                "timed_out": self.timed_out, "ok": self.done.is_set() and not self.error and not self.timed_out,
                **({"error": self.error} if self.error else {})}


def start(plan: dict, why: str, overlapped: bool, ws: int | None = None) -> tuple[Handle, bool]:
    """(the read, joined): a read in flight for this file is joined, never restarted; else a thread starts one."""
    path = plan["path"]
    with _lock:
        h = _current.get(path)
        if h is not None and not h.done.is_set():
            return h, True
        h = Handle(path, why, overlapped, os.path.getsize(path), ws, plan["model"])
        _current[path] = h
        threading.Thread(target=h.run, name="preread", daemon=True).start()
    return h, False


def inflight(path: str) -> Handle | None:
    with _lock:
        h = _current.get(path)
        return h if h is not None and not h.done.is_set() else None


# ---------------------------------------------------------------------- plan --
def plan(model: str, table, features=None, config_path: str | None = None) -> dict | None:
    """None when the tier table declares no pre-read for this model. Else {model, on, source, overlap, overlap_source,
    path, skipped?}: `skipped` says why nothing will be read (the switch is off, or the file cannot be found)."""
    if not getattr(table, "full", False) or not table.preread(model):
        return None
    sw = switches(features)
    out = {"model": model, **sw, "path": None}
    if not sw["on"]:
        out["skipped"] = f"switched off ({sw['source']}: YAMADORI_PREREAD / X-Yamadori-Features preread)"
        return out
    if os.environ.get("YAMADORI_OFFLINE_GUARD") == "1" and _READ is read_sequential:
        # an offline suite (scripts/run_tests.py) never reads a model file or a live process, whatever the table says
        # (max_mode._load's rule; found 2026-10-06: test_max_mode's swap on the live table would have read 39 GB)
        out["skipped"] = "not read: an offline suite (YAMADORI_OFFLINE_GUARD=1) never reads a model file"
        return out
    path, why = model_file(model, config_path)
    if path is None:
        out["skipped"] = why
    elif not os.path.isfile(path):
        out["skipped"] = f"the model file is not there: {os.path.basename(path)}"
    else:
        out["path"] = path
    return out


def available_bytes() -> int | None:
    """Memory available to the file cache now (free + standby: psutil's `available`, Windows' Available MBytes)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:                                                # noqa: BLE001
        return None


def decide_room(pl: dict) -> dict:
    """Does the cache have room for the whole file? {ok, available?, file_bytes, skipped?}. An unreadable figure does
    not block the read."""
    size = os.path.getsize(pl["path"])
    av = available_bytes()
    if av is not None and av < size:
        return {"ok": False, "available": av, "file_bytes": size,
                "skipped": f"only {av / 1e6:.0f} MB are available to the file cache, less than the file's {size / 1e6:.0f} "
                           "MB: a read would evict its own head (fn_first_prompt 20261006-proxy)"}
    return {"ok": True, "available": av, "file_bytes": size}


def decide_trimmed(pl: dict) -> dict:
    """The working-set rule for a LOADED model: {read: bool, ws?, file_bytes?, skipped?}."""
    path = pl["path"]
    size = os.path.getsize(path)
    ws, why = working_set(path)
    if ws is None:
        return {"read": False, "file_bytes": size, "skipped": f"the working set could not be read: {why}"}
    # TRIMMED MEANS FAR BELOW, not just below (coordinator, 2026-10-06): the measured states fall in two groups with a
    # wide gap -- trimmed 67 MB (4 h idle, n=1) and ~900 MB (`preread.py status` on the idle server), warm 39.1-44.4 GB
    # after a prompt (fn_first_prompt 20261006-a). "Below the file's size" put the warm 39.1 GB on the trimmed side, so a
    # warm agent loop would re-read 39 GB on ordinary turns. Half the file's size sits inside the measured gap with
    # ~19 GB either side; any cut in that gap reads the same data alike.
    if ws >= size // 2:
        return {"read": False, "ws": ws, "file_bytes": size,
                "skipped": f"working set {ws / 1e6:.0f} MB >= half the file's {size / 1e6:.0f} MB: not trimmed"}
    return {"read": True, "ws": ws, "file_bytes": size}


def skipped_record(pl: dict, why: str, reason: str, d: dict | None = None) -> dict:
    d = d or {}
    return {"why": why, "model": pl["model"], "skipped": reason, "file": os.path.basename(pl["path"] or "") or None,
            "file_bytes": d.get("file_bytes"),
            "working_set_mb": None if d.get("ws") is None else round(d["ws"] / 1e6),
            "available_mb": None if d.get("available") is None else round(d["available"] / 1e6)}


def _main(argv: list[str]) -> int:
    import json
    import tier_models
    if len(argv) < 2 or argv[1] != "status":
        print(__doc__.split("\n    python mcp/preread.py")[1].strip() if "\n    python" in __doc__ else "status")
        return 2
    model = argv[2] if len(argv) > 2 else "flash-next"
    t = tier_models.table()
    out: dict = {"model": model, "table": t.source, "declared": bool(getattr(t, "full", False) and t.preread(model))}
    path, why = model_file(model)
    out["file"], out["why_not"] = path, why
    if path and os.path.isfile(path):
        out["file_bytes"] = os.path.getsize(path)
        ws, w = working_set(path)
        out["working_set_bytes"], out["working_set_why_not"] = ws, w
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
