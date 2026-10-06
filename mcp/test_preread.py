#!/usr/bin/env python
"""The pre-read of Flash-Next's expert file (mcp/preread.py, wired in mcp/max_mode.py wait_ready), offline.

WHAT IS GATED
  1. WHERE THE FILE COMES FROM: the `-m` argument of the model's llama-swap entry, macros expanded, comments in the
     cmd ignored (a temp config.yaml); only a model whose tier-table ROW says `preread` has a plan (the live table:
     flash-next yes, bonsai and mirai-s no).
  2. THE SWAP: a flash-next swap starts the read in a thread and the request does not return before it ends (overlap,
     the default: the read is under way before the load ends; `preread_overlap` off: the read ends before the load
     starts); card_wait -- the flag proxy._TurnPump heartbeats on -- is up for the whole wait. A bonsai swap starts no
     thread.
  3. THE WORKING-SET RULE for a loaded model: below half the file's size -> read (why `trimmed`, working_set_mb recorded);
     at or above it, or unreadable -> no read, the reason recorded.
  4. THE LOCK: a read in flight for a file is joined, never restarted; reads of two files never overlap.
  5. THE SWITCH: YAMADORI_PREREAD=0 and X-Yamadori-Features {"preread": false} (the header wins over env), recorded.
  6. A MISSING FILE degrades to no read with a recorded reason; the swap still goes ahead. A cancelled request leaves
     the wait at once.
  7. x_yamadori.preread carries the record (proxy._x_yamadori).

NO NETWORK, NO LIVE FILES: llama-swap's /running and its load are stubbed, the model "file" is a few bytes in a temp
directory, the read is a stub that sleeps, the working set is a stub, every store is a temp path set before import.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_preread_")
os.environ["YAMADORI_MAX_STATE"] = os.path.join(_TMP, "max_mode.json")
os.environ.setdefault("YAMADORI_ACCOUNTS_DIR", os.path.join(_TMP, "accounts"))
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
for v in ("YAMADORI_MAX_MODEL", "YAMADORI_MAX_IDLE_S", "YAMADORI_PREREAD", "YAMADORI_PREREAD_OVERLAP"):
    os.environ.pop(v, None)

# A llama-swap config shaped like config.yaml's: macros, a cmd block with comment lines, a split-GGUF `-m`.
_MODELS_DIR = os.path.join(_TMP, "models").replace("\\", "/")
os.makedirs(_MODELS_DIR + "/flash-next/IQ2_XS", exist_ok=True)
SHARD1 = _MODELS_DIR + "/flash-next/IQ2_XS/X-00001-of-00002.gguf"
SHARD2 = _MODELS_DIR + "/flash-next/IQ2_XS/X-00002-of-00002.gguf"
SIZE = 3 * 1024 * 1024 + 17
with open(SHARD1, "wb") as f:
    f.write(b"\0" * SIZE)
with open(SHARD2, "wb") as f:
    f.write(b"\0" * 1000)
_CONFIG = os.path.join(_TMP, "config.yaml")
with open(_CONFIG, "w", encoding="utf-8") as f:
    f.write(f"""macros:
  base: "{_MODELS_DIR}"
  models: "${{base}}"
models:
  "flash-next":
    cmd: |
      server --port ${{PORT}}
      # a comment that mentions -m C:/nowhere/decoy.gguf
      -m ${{models}}/flash-next/IQ2_XS/X-00001-of-00002.gguf
      --spec-draft-model ${{models}}/flash-next/mtp.gguf -ngld 999
  "bonsai":
    cmd: |
      server --port ${{PORT}}
      -m ${{models}}/bonsai.gguf
""")
os.environ["YAMADORI_SWAP_CONFIG"] = _CONFIG

import cancel  # noqa: E402
import max_mode  # noqa: E402
import preread  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, str(detail)))
    return bool(ok)


EVENTS: list[str] = []
_EV_LOCK = threading.Lock()
_LOADED: set[str] = set()
READS: list[str] = []
LOADS: list[str] = []
_ACTIVE = {"n": 0, "peak": 0}
READ_S = 0.5
LOAD_S = 0.25
WS = {"v": 100_000_000, "why": ""}


def ev(s: str) -> None:
    with _EV_LOCK:
        EVENTS.append(s)


def fake_read(path, should_stop) -> int:
    with _EV_LOCK:
        _ACTIVE["n"] += 1
        _ACTIVE["peak"] = max(_ACTIVE["peak"], _ACTIVE["n"])
    READS.append(path)
    ev("read_start")
    try:
        t = time.time()
        while time.time() - t < READ_S and not should_stop():
            time.sleep(0.01)
        ev("read_end")
        return os.path.getsize(path)
    finally:
        with _EV_LOCK:
            _ACTIVE["n"] -= 1


def fake_load(model, timeout=0):
    LOADS.append(model)
    ev("load_start")
    time.sleep(LOAD_S)
    _LOADED.clear()
    _LOADED.add(model)
    ev("load_end")
    return True, "HTTP 200 (fake)"


def reload_stack(table_path: str | None = None):
    """max_mode on the live tier table (flash-next `preread: true`), llama-swap stubbed."""
    os.environ["YAMADORI_TIER_MODELS"] = table_path or os.path.join(HERE, "tier_models.yaml")
    importlib.reload(max_mode)
    max_mode.running = lambda: set(_LOADED)
    max_mode._load = fake_load
    max_mode._reset_for_tests()
    preread._READ = fake_read
    preread.working_set = lambda path: (WS["v"], WS["why"]) if WS["v"] is not None else (None, WS["why"] or "stub")
    preread._current.clear()
    preread._pid_cache.clear()
    EVENTS.clear()
    READS.clear()
    LOADS.clear()
    _LOADED.clear()
    _ACTIVE.update(n=0, peak=0)
    WS.update(v=100_000_000, why="")
    for v in ("YAMADORI_PREREAD", "YAMADORI_PREREAD_OVERLAP"):
        os.environ.pop(v, None)
    try:
        os.remove(os.environ["YAMADORI_MAX_STATE"])
    except OSError:
        pass
    return max_mode


def run_wait(model, features=None, watch=True):
    """wait_ready on a thread bound to a token; samples the token's card_wait flag while it runs.
    Returns (result, samples [(t, flag)], t_start, t_end)."""
    tok = cancel.Token()
    out: dict = {}
    samples: list[tuple[float, bool]] = []
    done = threading.Event()

    def work():
        with cancel.bound(tok):
            max_mode.set_current(model)
            try:
                out["r"] = max_mode.wait_ready(model, poll_s=0.02, features=features)
            except Exception as e:                                   # noqa: BLE001
                out["exc"] = e
        out["t1"] = time.time()
        done.set()
    t0 = time.time()
    th = threading.Thread(target=work)
    th.start()
    while not done.is_set():
        samples.append((time.time(), bool(getattr(tok, "card_wait", False))))
        time.sleep(0.01)
    th.join()
    return out, samples, t0, tok


def settle() -> None:
    """Let a finished read thread's bookkeeping end."""
    for _ in range(100):
        if not preread._current:
            return
        time.sleep(0.01)


# --------------------------------------------------------------------------------------------------------- tests
def test_file_and_plan():
    reload_stack()
    path, why = preread.model_file("flash-next")
    check(path == SHARD1 and why == "",
          "the file is the entry's -m, macros (nested) expanded, comment lines and --spec-draft-model ignored", path)
    check(preread.model_file("bonsai")[0] == _MODELS_DIR + "/bonsai.gguf", "another entry's own -m")
    p2, why2 = preread.model_file("nothing")
    check(p2 is None and "no cmd" in why2, "an entry the config lacks: no file, with the reason", why2)
    p3, why3 = preread.model_file("flash-next", os.path.join(_TMP, "absent.yaml"))
    check(p3 is None and "could not be read" in why3, "an unreadable config: no file, with the reason", why3)
    t = max_mode.TABLE
    check(t.preread("flash-next") and not t.preread("bonsai") and not t.preread("mirai-s"),
          "the live tier table declares the pre-read for flash-next only")
    check(preread.plan("bonsai", t) is None, "no plan for bonsai")
    pl = preread.plan("flash-next", t)
    check(pl and pl["path"] == SHARD1 and pl["on"] and pl["overlap"] and not pl.get("skipped"),
          "a plan for flash-next: shard 1, on, overlap by default", pl)
    check(preread.read_sequential(SHARD1, lambda: False) == SIZE, "read_sequential reads the whole file, once")
    n = [0]

    def stop_soon():
        n[0] += 1
        return n[0] > 1
    check(preread.read_sequential(SHARD1, stop_soon) < SIZE or SIZE <= preread.CHUNK,
          "read_sequential stops when asked (a file smaller than a chunk ends in one read)")


def test_swap_overlap_waits_with_heartbeats():
    reload_stack()
    out, samples, t0, tok = run_wait("flash-next")
    r = out["r"]
    pr = r.get("preread") or {}
    i = {k: EVENTS.index(k) for k in ("read_start", "load_start", "load_end", "read_end")}
    check(READS == [SHARD1] and LOADS == ["flash-next"], "a flash-next swap starts ONE read, of shard 1, and the load",
          json.dumps({"reads": READS, "loads": LOADS}))
    check(i["read_start"] < i["load_end"] and i["load_end"] < i["read_end"] and out["t1"] >= t0 + READ_S,
          "OVERLAP (default): the read is under way before the load ends, and the request does not return until "
          "the read has ended", json.dumps(EVENTS))
    check(pr.get("why") == "swap" and pr.get("overlapped") is True and pr.get("bytes") == SIZE
          and pr.get("file_bytes") == SIZE and pr.get("file") == os.path.basename(SHARD1) and pr.get("ok")
          and pr.get("joined") is False and pr.get("ms", 0) >= READ_S * 1000 * 0.9
          and "working_set_mb" in pr and pr.get("waited_ms", 0) > 0,
          "x_yamadori.preread's record: why swap, file, bytes, ms, overlapped, working_set_mb, joined, ok", pr)
    check((r.get("swap") or {}).get("load_s", 9) < READ_S and (r.get("swap") or {}).get("ok"),
          "load_s times the load alone (the read's wait is the record's waited_ms)", r.get("swap"))
    live = [f for t, f in samples if t0 + 0.05 < t < out["t1"] - 0.05]
    check(live and all(live) and not getattr(tok, "card_wait", True),
          "card_wait (the heartbeat flag) is up for the whole wait -- load and read -- and down after",
          f"{sum(live)}/{len(live)}")
    check("working_set_mb" in pr and pr["working_set_mb"] is None, "a swap records no working set (no server yet)")
    settle()


def test_swap_read_then_load():
    reload_stack()
    out, samples, t0, tok = run_wait("flash-next", features=json.dumps({"preread_overlap": False}))
    pr = out["r"].get("preread") or {}
    check(EVENTS == ["read_start", "read_end", "load_start", "load_end"] and pr.get("overlapped") is False,
          "preread_overlap off: the read ends before the load starts", json.dumps(EVENTS))
    live = [f for t, f in samples if t0 + 0.05 < t < out["t1"] - 0.05]
    check(live and all(live), "card_wait is up for the read and the load alike", f"{sum(live)}/{len(live)}")
    reload_stack()
    os.environ["YAMADORI_PREREAD_OVERLAP"] = "0"
    out, _s, _t, _k = run_wait("flash-next")
    check(EVENTS[:2] == ["read_start", "read_end"], "YAMADORI_PREREAD_OVERLAP=0 does the same", json.dumps(EVENTS))
    os.environ.pop("YAMADORI_PREREAD_OVERLAP", None)
    settle()


def test_bonsai_swap_starts_no_read():
    reload_stack()
    out, *_ = run_wait("bonsai")
    check(READS == [] and "preread" not in out["r"] and LOADS == ["bonsai"] and not preread._current,
          "a bonsai swap starts no read thread and records no preread", json.dumps({"reads": READS, "r": out["r"]}))
    _LOADED.clear()
    _LOADED.add("bonsai")
    out, *_ = run_wait("bonsai")
    check(READS == [] and "preread" not in out["r"], "a loaded bonsai is not checked either")


def test_working_set_rule():
    reload_stack()
    _LOADED.add("flash-next")
    WS.update(v=1_234_567, why="")   # the file is 3.1 MB
    out, samples, t0, tok = run_wait("flash-next")
    pr = out["r"].get("preread") or {}
    check(READS == [SHARD1] and not LOADS and pr.get("why") == "trimmed" and pr.get("working_set_mb") == 1
          and pr.get("overlapped") is False and pr.get("ok"),
          "a loaded flash-next whose working set is below the file's size: read (why trimmed, working_set_mb)", pr)
    live = [f for t, f in samples if t0 + 0.05 < t < out["t1"] - 0.05]
    check(live and all(live), "the trimmed wait holds card_wait too", f"{sum(live)}/{len(live)}")
    settle()
    reload_stack()
    _LOADED.add("flash-next")
    WS.update(v=SIZE, why="")
    out, *_ = run_wait("flash-next")
    pr = out["r"].get("preread") or {}
    check(READS == [] and pr.get("skipped") and "working set" in pr["skipped"] and pr.get("why") == "trimmed"
          and pr.get("working_set_mb") is not None,
          "a working set AT the file's size: no read, the reason and the numbers recorded", pr)
    WS.update(v=SIZE + 5_000_000_000, why="")
    out, *_ = run_wait("flash-next")
    check(READS == [], "a working set above it: no read")
    WS.update(v=SIZE - SIZE // 400, why="")   # the warm 39.1 GB of a 39.23 GB file, scaled: just under the size
    out, *_ = run_wait("flash-next")
    check(READS == [], "a WARM working set just under the file's size: no read (trimmed means below half the file)")
    WS.update(v=None, why="no llama-server process has this file in its command line")
    out, *_ = run_wait("flash-next")
    pr = out["r"].get("preread") or {}
    check(READS == [] and "could not be read" in (pr.get("skipped") or ""),
          "a working set that cannot be read: no read, the reason recorded", pr)
    check(not LOADS, "a loaded model is never loaded again by the check")


def test_lock_and_join():
    reload_stack()
    _LOADED.add("flash-next")
    WS.update(v=1_000_000, why="")
    res = []
    ths = [threading.Thread(target=lambda: res.append(run_wait("flash-next")[0]["r"])) for _ in range(3)]
    for t in ths:
        t.start()
        time.sleep(0.05)
    for t in ths:
        t.join()
    prs = [r.get("preread") or {} for r in res]
    check(len(READS) == 1 and sum(1 for p in prs if p.get("joined")) >= 1 and all(p.get("ok") for p in prs),
          "three requests, one read: the read in flight is JOINED, never restarted", json.dumps(
              {"reads": len(READS), "joined": [p.get("joined") for p in prs]}))
    settle()
    # two files: never at once
    reload_stack()
    pl1 = {"path": SHARD1, "model": "flash-next"}
    pl2 = {"path": SHARD2, "model": "flash-next"}
    h1, j1 = preread.start(pl1, "swap", True)
    h2, j2 = preread.start(pl2, "swap", True)
    h1b, j1b = preread.start(pl1, "swap", True)
    h1.wait(timeout=5)
    h2.wait(timeout=5)
    check(h1b is h1 and j1b and not j1 and not j2 and h1 is not h2 and _ACTIVE["peak"] == 1
          and len(READS) == 2, "a second file's read waits for the first (never two at once); the same file joins",
          json.dumps({"peak": _ACTIVE["peak"], "reads": len(READS)}))
    settle()
    check(not preread._current, "finished reads leave nothing registered")


def test_switch():
    reload_stack()
    out, *_ = run_wait("flash-next", features=json.dumps({"preread": False}))
    pr = out["r"].get("preread") or {}
    check(READS == [] and LOADS == ["flash-next"] and "switched off" in (pr.get("skipped") or "")
          and "header" in pr["skipped"],
          "X-Yamadori-Features {preread: false}: no read, the swap still goes ahead, the switch's source recorded", pr)
    reload_stack()
    os.environ["YAMADORI_PREREAD"] = "0"
    out, *_ = run_wait("flash-next")
    pr = out["r"].get("preread") or {}
    check(READS == [] and "env" in (pr.get("skipped") or ""), "YAMADORI_PREREAD=0: no read, source env", pr)
    _LOADED.clear()
    out, *_ = run_wait("flash-next", features=json.dumps({"preread": True}))
    check(READS == [SHARD1], "the header wins over the variable (forced on)", json.dumps(READS))
    os.environ.pop("YAMADORI_PREREAD", None)
    settle()
    import tiers
    sw = tiers.behaviours({})
    check(sw["preread"]["on"] and sw["preread_overlap"]["on"], "both switches default ON (x_yamadori.progress.switches)",
          json.dumps({k: sw[k] for k in ("preread", "preread_overlap")}))


def test_missing_file_degrades():
    reload_stack()
    cfg = os.path.join(_TMP, "config_missing.yaml")
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(f"macros:\n  models: \"{_MODELS_DIR}\"\nmodels:\n  flash-next:\n    cmd: |\n      server\n"
                f"      -m ${{models}}/flash-next/gone.gguf\n")
    os.environ["YAMADORI_SWAP_CONFIG"] = cfg
    try:
        out, *_ = run_wait("flash-next")
    finally:
        os.environ["YAMADORI_SWAP_CONFIG"] = _CONFIG
    pr = out["r"].get("preread") or {}
    check("exc" not in out and READS == [] and LOADS == ["flash-next"] and "not there" in (pr.get("skipped") or "")
          and (out["r"].get("swap") or {}).get("ok"),
          "a missing file: no read, the reason recorded, the swap still happens", pr)
    # a file that vanishes between the plan and the start
    reload_stack()
    real = preread.start

    def boom(*a, **k):
        raise FileNotFoundError("gone")
    preread.start = boom
    try:
        out, *_ = run_wait("flash-next")
    finally:
        preread.start = real
    check("exc" not in out and LOADS == ["flash-next"], "a read that cannot start never fails the request")
    # a read that raises
    reload_stack()

    def bad_read(path, should_stop):
        raise OSError("disk error")
    preread._READ = bad_read
    out, *_ = run_wait("flash-next")
    pr = out["r"].get("preread") or {}
    check("exc" not in out and pr.get("ok") is False and "disk error" in pr.get("error", ""),
          "a read that fails is recorded (ok false, error) and the request goes on", pr)
    settle()


def test_cancel_leaves_the_wait():
    reload_stack()
    _LOADED.add("flash-next")
    WS.update(v=1_000_000, why="")
    tok = cancel.Token()
    out: dict = {}

    def work():
        with cancel.bound(tok):
            max_mode.set_current("flash-next")
            try:
                max_mode.wait_ready("flash-next", poll_s=0.02)
            except cancel.Cancelled as e:
                out["cancelled"] = e
    th = threading.Thread(target=work)
    t0 = time.time()
    th.start()
    time.sleep(0.15)
    tok.cancel("test")
    th.join(timeout=5)
    check(out.get("cancelled") is not None and time.time() - t0 < READ_S + 0.3 and not getattr(tok, "card_wait", True),
          "a cancelled request leaves the wait at once and clears card_wait (the read goes on without it)")
    settle()


def test_timeout_bound():
    reload_stack()
    saved = preread.TIMEOUT_S
    preread.TIMEOUT_S = 0.2
    try:
        pl = {"path": SHARD1, "model": "flash-next"}

        def slow(path, should_stop):
            while not should_stop():
                time.sleep(0.01)
            return 10
        preread._READ = slow
        h, _ = preread.start(pl, "swap", True)
        got = h.wait(timeout=5)
        rec = h.record(False, 0.0)
        check(got and rec["timed_out"] and not rec["ok"] and rec["bytes"] == 10,
              "a read past TIMEOUT_S (llama-swap's 900 s) stops and says timed_out", rec)
    finally:
        preread.TIMEOUT_S = saved
    settle()


def test_offline_guard_never_reads_a_real_file():
    """Found 2026-10-06: test_max_mode swaps flash-next on the live table, whose config.yaml names a 39 GB file that
    exists on this machine. Under the offline guard the real reader plans nothing."""
    reload_stack()
    preread._READ = preread.read_sequential
    pl = preread.plan("flash-next", max_mode.TABLE)
    check(pl and "offline suite" in (pl.get("skipped") or "") and pl.get("path") is None,
          "under YAMADORI_OFFLINE_GUARD=1 the real reader is never planned (no model file, no live process)", pl)
    out, *_ = run_wait("flash-next")
    check(READS == [] and LOADS == ["flash-next"] and "offline suite" in (out["r"].get("preread") or {}).get("skipped", ""),
          "a swap on the live table under the guard goes ahead with the skip recorded", out["r"].get("preread"))


def test_x_yamadori_carries_it():
    import proxy
    rec = {"why": "swap", "file": "x.gguf", "bytes": 1, "ms": 2, "overlapped": True, "working_set_mb": None}
    try:
        x = proxy._x_yamadori({"_preread": rec, "_capacity": {"waited_s": 0.0}}, hops=0)
        check(x.get("preread") == rec and "preread" not in (x.get("capacity") or {}),
              "x_yamadori.preread carries the record beside capacity", json.dumps(x.get("preread")))
        x = proxy._x_yamadori({}, hops=0)
        check("preread" not in x, "absent when there was none")
    except Exception as e:                                           # noqa: BLE001
        import traceback
        check(False, "proxy._x_yamadori with a preread", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
    src = open(os.path.join(HERE, "proxy.py"), encoding="utf-8").read()
    check('max_mode.wait_ready(body["_upstream_model"], features=body.get("_features"))' in src
          and 'waited.pop("preread", None)' in src,
          "proxy._run_turn passes the request's features to wait_ready and moves the record to _preread")


def main() -> int:
    for fn in (test_file_and_plan, test_swap_overlap_waits_with_heartbeats, test_swap_read_then_load,
               test_bonsai_swap_starts_no_read, test_working_set_rule, test_lock_and_join, test_switch,
               test_missing_file_degrades, test_cancel_leaves_the_wait, test_timeout_bound,
               test_offline_guard_never_reads_a_real_file, test_x_yamadori_carries_it):
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
            import traceback
            check(False, f"{fn.__name__} raised", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    os.environ.pop("YAMADORI_TIER_MODELS", None)
    bad = [r for r in _results if not r[0]]
    for ok, name, detail in _results:
        print(("ok    " if ok else "FAIL  ") + name + (f"  ({detail[:300]})" if not ok and detail else ""))
    print(f"\n{'=' * 70}\n  {len(_results) - len(bad)}/{len(_results)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
