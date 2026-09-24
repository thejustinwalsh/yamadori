#!/usr/bin/env python
"""Gated experiment: the main model at q4_0 K/V + K mean-centering and a larger
-c, against today's q8_0 at 163,840. docs/CONTEXT-EXPANSION.md is the plan;
the gates it states are the GATES dict below, written into the run's
manifest.json BEFORE the first request and read back from there by
analyse.py, so they cannot be moved after the data is seen.

    python bench/kv_context/run.py --plan                      # sends nothing
    python bench/kv_context/run.py --window --key-file K --run-id kvq4-20260924
    python bench/kv_context/run.py --restore                   # production back

Queue it (it waits for an idle card, then runs; bench/queue_runner.py):

    python bench/queue_runner.py add kv_context/run.py --needs model,proxy \
        --args "--window --key-file C:/path/key --run-id kvq4-20260924 --wait-idle 720"

WHAT IT TOUCHES, AND WHY THAT NEEDS --window. The 5060 Ti holds one main
model. To measure a trial entry the production `bonsai` must be unloaded, and
while a `context-trial-*` profile is active every request for `bonsai` --
the operator's, Hermes', the worker's -- is served by the trial. So the run
refuses to start without --window (the operator's statement that the window
is theirs to give), pauses the worker's gpu lane, and ALWAYS puts production
back in `finally`: profile off, trial unloaded, `bonsai` reloaded and checked
at -c 163840. `--restore` does only that.

WHERE REQUESTS GO. Every generation goes through the proxy on :1234, the door
users use (AGENTS.md). The proxy has no per-request way to reach another
upstream model (catalog.resolve maps every public name to `bonsai`), so the
trial is selected one layer down with a llama-swap profile pin
(config.yaml `profiles:`): the proxy keeps asking for `bonsai` and llama-swap
serves the pinned entry. The control plane -- load, unload, profile, /running,
and erasing idle slot caches before a long haystack -- is llama-swap's own API
on :11434, because the proxy exposes none of it. No request is sent to a
llama-server port directly.

PHASES (each resumable: rows are keyed, a stack error is never "done")

  fit       per candidate -c (262144, 196608): unload bonsai, activate the
            profile, load, read VRAM idle, then fill the pool through :1234
            (a ~55% prompt then a ~30% prompt, idle slots erased first) while
            polling VRAM every second. Under the floor at idle or at peak ->
            unloaded at once, FAIL, no further phase uses it. The largest
            passing candidate is the q4 arm.
  speed     per arm (q8 = production, q4 = that candidate): a ~2k and a ~98k
            prompt, 3 reps, a cold `code` request then a cached `prose`
            request. Wall clock through the proxy (it drops llama-server's
            timings): prefill = prompt_tokens / TTFT, decode = completion
            tokens / streaming time. Thinking off (tier `minimal`).
  accuracy  needles in a SYNTHETIC codebase (synth_code.py; the probe and
            grader are bench/longctx/haystack.py). Same items, same seed, both
            arms; shared rungs up to 151,552 (fits q8's pool), then q4-only
            rungs up to near its window. Thinking off, forced-off features.
  quality   LiveBench coding, the 21 questions of lb-20260923-minp0, bare
            @ medium with check_code and repair forced off, via
            bench/livebench/run_arm.sh in WSL: a fresh q8 control arm, then
            the q4 arm. Needs arms `kv-q8` and `kv-q4` in drive.py ARMS (it
            says exactly what to add if they are missing, and skips).

A VRAM guard polls the card every second during every trial phase; below
ABORT_FREE_MIB it unloads the trial and the phase stops as an outage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "results")
LONGCTX = os.path.join(ROOT, "bench", "longctx")
LIVEBENCH = os.path.join(ROOT, "bench", "livebench")
CONFIG = os.path.join(ROOT, "config.yaml")
sys.path.insert(0, HERE)
sys.path.insert(0, LONGCTX)

import haystack as hs  # noqa: E402
import synth_code  # noqa: E402
import vram  # noqa: E402

PROXY = os.environ.get("YAMADORI_PROXY_URL", "http://127.0.0.1:1234")
SWAP = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
CARD_UUID = "GPU-de660e90-0e9c-d465-b389-6df63021b920"      # RTX 5060 Ti
PROD_ID, PROD_C = "bonsai", 163_840
MODELS_DIR = "C:/Users/jwals/textgen/user_data/models"
BIAS = f"{MODELS_DIR}/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-mtp-lean.kv-mean-center-q4_0.gguf"
CANDIDATES = {262_144: ("bonsai-q4kv", "context-trial-262k"),
              196_608: ("bonsai-q4kv-196k", "context-trial-196k")}
EMPTY_CARD_MIB = 900          # "bonsai is gone" once the card uses less than this
ABORT_FREE_MIB = 512          # the guard's hard stop, far below any floor
LOAD_TIMEOUT_S = 900          # config.yaml healthCheckTimeout
REQ_TIMEOUT_S = 3600          # a ~250k prefill at ~200 tok/s is ~21 min
FIT_MARGIN = 1.03             # calibration error allowance (bench/longctx)
PROXY_GEN = 2048              # tiers.A_MIN: the answer room the proxy reserves
SYNTH_SEED = 7
ITEM_SEED = 1
FEATURES = {"retrieval": False, "hints": False, "investigate": False,
            "fanout": 1, "check_code": False, "repair": False}
SPEED_PROMPTS = {
    "code": ("Write a new TypeScript module that uses the APIs defined in the "
             "code above: one class with three methods and full type "
             "annotations. Output only code. Keep it under 250 words."),
    "prose": ("Summarise, module by module, what the code above defines and how "
              "the pieces fit together. Plain prose, under 250 words."),
}
SHARED_LADDER = [8_192, 32_768, 65_536, 98_304, 131_072, 151_552]
BEYOND_LADDER = {262_144: [196_608, 229_376, 249_856], 196_608: [180_224]}
SPEED_LENGTHS = [2_048, 98_304]

# PRE-REGISTERED (docs/CONTEXT-EXPANSION.md s.5). Copied into manifest.json at
# the start of a run; analyse.py reads the manifest's copy.
GATES = {
    "floor_mib": 3_072,
    "speed_min_ratio": 0.85,
    "acc_max_drop_points": 5.0,
    "acc_alpha": 0.05,
    "quality_alpha": 0.05,
    "quality_max_drop_questions": 2,
}

_EVENTS: str | None = None


# ------------------------------------------------------------- plumbing ----
def log(msg: str, **kw) -> None:
    print(f"  {time.strftime('%H:%M:%S')} {msg}", flush=True)
    if _EVENTS:
        with open(_EVENTS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "msg": msg, **kw}) + "\n")


def load_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    return out


def done_keys(path: str) -> set[str]:
    return {r["key"] for r in load_rows(path)
            if r.get("key") and r.get("outcome") not in ("stack_error", None)}


def append(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _longctx_run():
    """bench/longctx/run.py as a module (its Client and speed_of), loaded by
    path because this file is also called run.py."""
    spec = importlib.util.spec_from_file_location(
        "longctx_run", os.path.join(LONGCTX, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- gpu ---------
def gpu_read() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", CARD_UUID],
            capture_output=True, text=True, timeout=15)
        used, free, util = [int(x.strip()) for x in out.stdout.strip().split(",")]
        return {"used": used, "free": free, "util": util, "t": time.time()}
    except Exception:                                            # noqa: BLE001
        return None


def gpu_settle(n: int = 5, every: float = 2.0) -> dict | None:
    xs = []
    for _ in range(n):
        g = gpu_read()
        if g:
            xs.append(g)
        time.sleep(every)
    if not xs:
        return None
    return {"used": int(statistics.median(x["used"] for x in xs)),
            "free": int(statistics.median(x["free"] for x in xs)),
            "min_free": min(x["free"] for x in xs), "n": len(xs)}


class VramGuard(threading.Thread):
    """Polls the card every second: keeps the peak and the minimum free, and
    below ABORT_FREE_MIB unloads the trial once and raises the flag."""

    def __init__(self, on_abort=None, every: float = 1.0):
        super().__init__(daemon=True)
        self.on_abort, self.every = on_abort, every
        self.max_used: int | None = None
        self.min_free: int | None = None
        self.aborted = threading.Event()
        self._stop = threading.Event()

    def reset(self) -> None:
        self.max_used, self.min_free = None, None

    def run(self):
        while not self._stop.is_set():
            g = gpu_read()
            if g:
                self.max_used = g["used"] if self.max_used is None else max(self.max_used, g["used"])
                self.min_free = g["free"] if self.min_free is None else min(self.min_free, g["free"])
                if g["free"] < ABORT_FREE_MIB and not self.aborted.is_set():
                    self.aborted.set()
                    log(f"VRAM GUARD: {g['free']} MiB free < {ABORT_FREE_MIB}; unloading the trial",
                        kind="guard", gpu=g)
                    if self.on_abort:
                        try:
                            self.on_abort()
                        except Exception as e:                   # noqa: BLE001
                            log(f"guard unload failed: {e}")
            self._stop.wait(self.every)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=10)


# ------------------------------------------------------------- llama-swap --
def http(method: str, url: str, body: dict | None = None,
         timeout: float = 30) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception:                                        # noqa: BLE001
            return e.code, ""
    except Exception as e:                                       # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def running() -> list[dict]:
    st, txt = http("GET", f"{SWAP}/running", timeout=10)
    if st != 200:
        return []
    try:
        return json.loads(txt).get("running") or []
    except ValueError:
        return []


def profiles() -> dict | None:
    st, txt = http("GET", f"{SWAP}/api/profiles", timeout=10)
    if st != 200:
        return None
    try:
        return json.loads(txt)
    except ValueError:
        return None


def set_profile(name: str | None) -> None:
    st, txt = http("PUT", f"{SWAP}/api/profiles/active", {"name": name})
    if st != 200:
        raise RuntimeError(f"profile {name!r}: HTTP {st} {txt[:200]}")
    log(f"llama-swap profile -> {name}", kind="profile", profile=name)


def unload(model_id: str) -> None:
    st, txt = http("POST", f"{SWAP}/api/models/unload/{model_id}", timeout=120)
    log(f"unload {model_id}: HTTP {st}", kind="unload", model=model_id, status=st)


def running_ids() -> dict[str, dict]:
    return {m.get("model"): m for m in running() if m.get("model")}


def wait_card_below(mib: int, timeout: float = 180) -> dict | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        g = gpu_read()
        if g and g["used"] < mib:
            return g
        time.sleep(2)
    return None


def load_via_health(timeout: float = LOAD_TIMEOUT_S) -> float:
    """GET /upstream/bonsai/health until 200. With a profile active the pin
    sends it to the trial entry; llama-swap starts whichever it resolves to."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        st, txt = http("GET", f"{SWAP}/upstream/{PROD_ID}/health", timeout=timeout)
        last = (st, txt[:200])
        if st == 200:
            return time.time() - t0
        time.sleep(3)
    raise RuntimeError(f"load did not answer 200 in {timeout}s: {last}")


def erase_idle_slots() -> int:
    """Free the KV cells idle slots still hold, so one long haystack can use
    the whole unified pool. Control plane, through llama-swap's upstream
    passthrough (the profile pin applies). Returns slots erased."""
    st, txt = http("GET", f"{SWAP}/upstream/{PROD_ID}/slots", timeout=30)
    if st != 200:
        return 0
    try:
        slots = json.loads(txt)
    except ValueError:
        return 0
    n = 0
    for s in slots if isinstance(slots, list) else []:
        if s.get("is_processing"):
            continue
        sid = s.get("id")
        st2, _ = http("POST", f"{SWAP}/upstream/{PROD_ID}/slots/{sid}?action=erase", {}, timeout=30)
        n += st2 == 200
    return n


def ensure_production() -> dict:
    """Profile off, trials unloaded, bonsai loaded and checked at -c 163840."""
    try:
        set_profile(None)
    except RuntimeError as e:
        log(f"profile reset failed ({e}); continuing to unload trials")
    ids = running_ids()
    for mid, _ in CANDIDATES.values():
        if mid in ids:
            unload(mid)
    if PROD_ID not in running_ids():
        wait_card_below(EMPTY_CARD_MIB, timeout=180)
        secs = load_via_health()
        log(f"production {PROD_ID} reloaded in {secs:.1f}s", kind="load", model=PROD_ID, seconds=secs)
    m = running_ids().get(PROD_ID) or {}
    cmd = m.get("cmd") or ""
    ok = f"-c {PROD_C}" in cmd and "--cache-type-k q8_0" in cmd
    rec = {"model": PROD_ID, "state": m.get("state"), "cmd_ok": ok, "gpu": gpu_settle(3, 1.0)}
    log(f"production check: {rec}", kind="production", **rec)
    if not ok:
        raise RuntimeError(f"production {PROD_ID} is not the -c {PROD_C} q8_0 entry: {cmd[:300]}")
    return rec


def ensure_trial(c: int) -> dict:
    """Unload bonsai, activate the candidate's profile, load it, verify it."""
    mid, prof = CANDIDATES[c]
    ids = running_ids()
    if mid in ids and (profiles() or {}).get("active") == prof:
        return {"model": mid, "already": True}
    set_profile(None)
    for other in [PROD_ID] + [m for m, _ in CANDIDATES.values()]:
        if other in running_ids():
            unload(other)
    empty = wait_card_below(EMPTY_CARD_MIB, timeout=180)
    if not empty:
        raise RuntimeError(f"card did not empty below {EMPTY_CARD_MIB} MiB after unloading")
    set_profile(prof)
    secs = load_via_health()
    m = running_ids().get(mid) or {}
    cmd = m.get("cmd") or ""
    ok = (f"-c {c}" in cmd and "--cache-type-k q4_0" in cmd
          and "--kv-mean-center" in cmd and PROD_ID not in running_ids())
    rec = {"model": mid, "profile": prof, "load_seconds": round(secs, 1),
           "empty_card_used_mib": empty["used"], "cmd_ok": ok}
    log(f"trial {mid} loaded in {secs:.1f}s", kind="load", **rec)
    if not ok:
        raise RuntimeError(f"expected {mid} alone with -c {c} q4_0 + bias; running: "
                           f"{list(running_ids())}; cmd: {cmd[:300]}")
    return rec


# ------------------------------------------------------------- proxy -------
class Proxy:
    """bench/longctx's Client, pointed at :1234, with this run's headers."""

    def __init__(self, key: str, run_id: str):
        lc = _longctx_run()
        self.lc = lc
        self.client = lc.Client(PROXY, key, dict(FEATURES), timeout=REQ_TIMEOUT_S)
        self.run_id = run_id
        base_headers = self.client._headers

        def headers(stream: bool = False, _b=base_headers, _self=self):
            h = _b(stream)
            if _self.session:
                h["X-Yamadori-Session"] = _self.session
            return h
        self.session = ""
        self.client._headers = headers

    def ask(self, msgs: list[dict], session: str, max_tokens: int = 256) -> dict:
        self.session = "kv-" + hashlib.sha1(f"{self.run_id}|{session}".encode()).hexdigest()[:24]
        body = {"model": "yamadori", "messages": msgs, "reasoning_effort": "minimal",
                "max_tokens": max_tokens}
        return self.client.chat(body, stream=True)

    def speed_of(self, res: dict) -> dict:
        return self.lc.speed_of(res)


def calibrate(px: Proxy, chunks) -> dict:
    """Prompt tokens of the wrapper alone and chars/token of the synthetic
    filler, from the proxy's usage on two requests."""
    it = hs.make_item(ITEM_SEED, 0)
    q = hs.question(it, "multi")
    pts = {}
    for n in (0, 48_000):
        text = hs.filler(ITEM_SEED, 0, n, chunks) if n else ""
        res = px.ask(hs.messages(text, q, f"cal{n}-{time.time_ns()}"), f"cal{n}", 16)
        pt = (res.get("usage") or {}).get("prompt_tokens")
        if not res.get("ok") or not pt:
            raise RuntimeError(f"calibration failed: {res.get('error_kind')} {res.get('error')}")
        pts[n] = (len(text), pt)
    (c0, t0), (c1, t1) = pts[0], pts[48_000]
    return {"overhead_tokens": t0, "chars_per_token": (c1 - c0) / max(t1 - t0, 1),
            "method": "two requests through :1234, usage.prompt_tokens", "at": time.time()}


def chars_for(L: int, cal: dict) -> int:
    return max(int((L - cal["overhead_tokens"]) * cal["chars_per_token"]), 1000)


def fits(L: int, pool: int) -> bool:
    return L * FIT_MARGIN + PROXY_GEN <= pool


# ------------------------------------------------------------- phases ------
def phase_fit(px: Proxy, man: dict, run_dir: str, guard: VramGuard, floor: int,
              chunks) -> None:
    path = os.path.join(run_dir, "fit.jsonl")
    done = {r["c"]: r for r in load_rows(path) if r.get("verdict") in ("pass", "fail")}
    for c in sorted(CANDIDATES, reverse=True):
        if c in done:
            log(f"fit {c}: already {done[c]['verdict']}")
            continue
        row = {"key": f"fit|{c}", "c": c, "model": CANDIDATES[c][0], "floor_mib": floor,
               "projected": vram.project(c, conservative=True), "t": time.time()}
        try:
            row["load"] = ensure_trial(c)
        except Exception as e:                                   # noqa: BLE001
            row.update(verdict="fail", why=f"load: {e}", outcome="measured")
            append(path, row)
            log(f"fit {c}: FAIL at load: {e}")
            unload(CANDIDATES[c][0])
            continue
        idle = gpu_settle(5, 2.0)
        row["idle"] = idle
        if not idle or idle["min_free"] < floor:
            row.update(verdict="fail", outcome="measured",
                       why=f"idle free {idle and idle['min_free']} MiB < floor {floor}")
            append(path, row)
            log(f"fit {c}: FAIL idle: {row['why']}")
            unload(CANDIDATES[c][0])
            continue
        # Stress: fill the unified pool through the proxy.
        guard.reset()
        erased = erase_idle_slots()
        stress = []
        # 55% then 30% of the pool: the first prompt's cells stay cached in its
        # slot while the second runs, so together they hold ~85% of the pool
        # plus two answer allowances, with room for the 3% calibration error.
        for frac, tag in ((0.55, "main-like"), (0.30, "helper-like")):
            L = int(c * frac)
            text = hs.filler(ITEM_SEED, 90_000 + len(stress), chars_for(L, man["calibration"]), chunks)
            nonce = hashlib.sha1(f"{run_dir}|fit|{c}|{tag}|{time.time_ns()}".encode()).hexdigest()[:16]
            res = px.ask(hs.messages(text, SPEED_PROMPTS["prose"], nonce), f"fit-{c}-{tag}", 128)
            stress.append({"tag": tag, "target_L": L, "ok": bool(res.get("ok")),
                           "error_kind": res.get("error_kind"),
                           "prompt_tokens": (res.get("usage") or {}).get("prompt_tokens"),
                           "seconds": round(res.get("seconds") or 0, 1),
                           "finish_reason": res.get("finish_reason")})
            log(f"fit {c} stress {tag}: {stress[-1]}")
            if guard.aborted.is_set():
                break
        after = gpu_settle(3, 2.0)
        row.update(stress=stress, slots_erased=erased, after=after,
                   peak_used_mib=guard.max_used, min_free_mib=guard.min_free)
        bad = [s for s in stress if not s["ok"]]
        if guard.aborted.is_set():
            row.update(verdict="fail", why="VRAM guard tripped during stress")
        elif bad:
            row.update(verdict="fail", why=f"stress request failed: {bad[0]['error_kind']}")
        elif guard.min_free is None or guard.min_free < floor:
            row.update(verdict="fail", why=f"peak free {guard.min_free} MiB < floor {floor}")
        else:
            row.update(verdict="pass", why=f"idle {idle['min_free']} / peak {guard.min_free} MiB free >= {floor}")
        row["outcome"] = "measured"
        append(path, row)
        log(f"fit {c}: {row['verdict'].upper()} -- {row['why']}")
        if row["verdict"] == "fail":
            unload(CANDIDATES[c][0])
        guard.aborted.clear()


def adopted_candidate(run_dir: str) -> int | None:
    rows = [r for r in load_rows(os.path.join(run_dir, "fit.jsonl")) if r.get("verdict") == "pass"]
    return max((r["c"] for r in rows), default=None)


def arm_state(arm: str, c: int | None) -> dict:
    return ensure_production() if arm == "q8" else ensure_trial(c)


def phase_speed(px: Proxy, man: dict, run_dir: str, arm: str, pool: int, chunks,
                reps: int, guard: VramGuard, deadline: float | None) -> str | None:
    path = os.path.join(run_dir, "speed.jsonl")
    done = done_keys(path)
    errs = 0
    for L in SPEED_LENGTHS:
        if not fits(L, pool):
            continue
        for rep in range(reps):
            nonce = None
            for content in ("code", "prose"):
                fresh = False
                key = f"speed|{arm}|{L}|{rep}|{content}"
                if key in done:
                    continue
                if deadline and time.time() > deadline:
                    return "deadline"
                if guard.aborted.is_set():
                    return "vram guard"
                if content == "code" or nonce is None:
                    nonce = hashlib.sha1(f"{key}|{time.time_ns()}".encode()).hexdigest()[:16]
                    fresh = True
                    # Unified KV: cells an idle slot still caches are not
                    # handed to a new long prompt reliably, so free them.
                    erase_idle_slots()
                text = hs.filler(ITEM_SEED, 50_000 + rep, chars_for(L, man["calibration"]), chunks)
                res = px.ask(hs.messages(text, SPEED_PROMPTS[content], nonce),
                             f"speed-{arm}-{L}-{rep}", 1024)
                row = {"key": key, "arm": arm, "pool": pool, "L": L, "rep": rep,
                       "content": content, "cold": fresh, "t": time.time(),
                       "gpu_after": gpu_read()}
                if not res.get("ok"):
                    row.update(outcome="stack_error", error_kind=res.get("error_kind"),
                               error=(res.get("error") or "")[:300])
                    errs += 1
                else:
                    errs = 0
                    row.update(outcome="measured", finish_reason=res.get("finish_reason"),
                               usage=res.get("usage"), **px.speed_of(res))
                append(path, row)
                log(f"speed {arm} L={L} rep={rep} {content}: {row.get('outcome')} "
                    f"prefill={row.get('prefill_tps')} decode={row.get('decode_tps')}")
                if errs >= 3:
                    return "3 consecutive stack errors -- an outage, not data"
    return None


def phase_accuracy(px: Proxy, man: dict, run_dir: str, arm: str, pool: int, ladder: list[int],
                   items: int, chunks, guard: VramGuard, deadline: float | None) -> str | None:
    path = os.path.join(run_dir, "accuracy.jsonl")
    done = done_keys(path)
    errs = 0
    for L in ladder:
        if not fits(L, pool):
            log(f"accuracy {arm}: {L} does not fit pool {pool}; ladder ends")
            break
        for i in range(items):
            keys = {t: f"acc|{arm}|{L}|{i}|{t}" for t in hs.TASKS}
            if all(k in done for k in keys.values()):
                continue
            if deadline and time.time() > deadline:
                return "deadline"
            if guard.aborted.is_set():
                return "vram guard"
            it = hs.make_item(ITEM_SEED, i)
            h = hs.build_haystack(it, chars_for(L, man["calibration"]), chunks)
            nonce = hashlib.sha1(f"{ITEM_SEED}|{i}|{L}".encode()).hexdigest()[:16]
            erase_idle_slots()     # the new haystack gets the whole pool
            for task in hs.TASKS:
                if keys[task] in done:
                    continue
                res = px.ask(hs.messages(h["text"], hs.question(it, task), nonce),
                             f"acc-{arm}-{L}-{i}", 256)
                row = {"key": keys[task], "arm": arm, "pool": pool, "L": L, "item": i,
                       "task": task, "t": time.time(), "haystack_chars": h["chars"]}
                if not res.get("ok"):
                    row.update(outcome="stack_error", error_kind=res.get("error_kind"),
                               error=(res.get("error") or "")[:300])
                    errs += 1
                else:
                    errs = 0
                    g = hs.grade(it, task, res.get("content"), res.get("finish_reason"))
                    row.update(outcome=g["outcome"], correct=g["correct"], parse=g["parse"],
                               got=g["got"], expected=hs.expected(it, task),
                               finish_reason=res.get("finish_reason"),
                               prompt_tokens=(res.get("usage") or {}).get("prompt_tokens"),
                               seconds=round(res.get("seconds") or 0, 1),
                               content=(res.get("content") or "")[-600:])
                    if "per_needle" in g:
                        depth = {n["name"]: n["depth"] for n in it["needles"]}
                        row["per_needle"] = [{"depth": depth[k], "ok": v}
                                             for k, v in g["per_needle"].items()]
                append(path, row)
                log(f"acc {arm} L={L} item={i} {task}: {row['outcome']}")
                if errs >= 3:
                    return "3 consecutive stack errors -- an outage, not data"
    return None


# LiveBench -------------------------------------------------------------------
LB_SOURCE_RUN = "lb-20260923-minp0"
ARMS_SNIPPET = '''    # docs/CONTEXT-EXPANSION.md: the KV trial's paired arms. bonsai's header
    # plus check_code/repair forced OFF, matching the cached lb-20260923-minp0
    # bonsai answers (produced before check_code existed). Same header, two
    # display names, so a q8 and a q4 answer never share an answer file.
    "kv-q8": {
        "display": "yamadori-kv-q8-arm",
        "features": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1,
                     "effort": "medium", "check_code": False, "repair": False},
    },
    "kv-q4": {
        "display": "yamadori-kv-q4-arm",
        "features": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1,
                     "effort": "medium", "check_code": False, "repair": False},
    },'''


def livebench_arms_missing() -> list[str]:
    sys.path.insert(0, LIVEBENCH)
    try:
        import drive  # noqa: F401
        return [a for a in ("kv-q8", "kv-q4") if a not in drive.ARMS]
    except Exception as e:                                       # noqa: BLE001
        return [f"(drive.py did not import: {e})"]


def to_wsl(path: str) -> str:
    p = os.path.abspath(path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    return f"/mnt/{m.group(1).lower()}/{m.group(2)}" if m else p


def phase_quality(run_dir: str, arm: str, key_file: str, distro: str | None,
                  lb_api_base: str | None) -> str | None:
    missing = livebench_arms_missing()
    if missing:
        msg = ("quality NOT RUN: bench/livebench/drive.py ARMS lacks " + ", ".join(missing)
               + ". Situation: LiveBench stores answers per display name, so a second "
               "run of arm `bonsai` would find the cached q8 answers and skip every "
               "question. Retryable: yes, once the arms exist. Remedy (operator): add "
               "to drive.py ARMS:\n" + ARMS_SNIPPET)
        log(msg, kind="quality_skipped")
        return msg
    lb_run = os.path.basename(run_dir) + "-lb"
    lb_dir = os.path.join(LIVEBENCH, "results", lb_run)
    os.makedirs(os.path.join(lb_dir, "logs"), exist_ok=True)
    order = os.path.join(lb_dir, "order_coding.json")
    if not os.path.exists(order):
        shutil.copyfile(os.path.join(LIVEBENCH, "results", LB_SOURCE_RUN, "order_coding.json"), order)
    lb_arm = f"kv-{arm}"
    env = f"KEY_FILE='{to_wsl(key_file)}' " + (f"LB_API_BASE='{lb_api_base}' " if lb_api_base else "")
    here = to_wsl(LIVEBENCH)
    cmds = [f"{env}bash {here}/run_arm.sh {lb_arm} coding {lb_run} 21",
            f"cd {here} && . ~/livebench-run/.venv/bin/activate && "
            f"HF_HUB_DISABLE_PROGRESS_BARS=1 python score.py --run-dir results/{lb_run} "
            f"--arm {lb_arm} --category coding"]
    for c in cmds:
        argv = ["wsl.exe"] + (["-d", distro] if distro else []) + ["--", "bash", "-lc", c]
        log(f"quality {arm}: {c.replace(env, '<env> ')}", kind="quality_cmd")
        p = subprocess.run(argv, text=True)
        if p.returncode != 0:
            return f"quality {arm}: command failed rc={p.returncode}"
    rows = os.path.join(lb_dir, f"rows_{lb_arm}_coding.jsonl")
    n = len(load_rows(rows))
    log(f"quality {arm}: {n} scored rows in {os.path.relpath(rows, ROOT)}", kind="quality_done")
    return None if n else f"quality {arm}: no scored rows"


# ------------------------------------------------------------- gates -------
def busy_elsewhere(me_pid: int) -> list[str]:
    """Benchmark processes other than this one and its ancestors (the queue
    runner that launched it must not count as busy)."""
    pats = ("bench/domain/run.py", "bench\\domain\\run.py", "livecodebench.py",
            "swebench", "swe_bench", "bench/longctx/run.py", "bench\\longctx\\run.py",
            "drive.py", "recipe_oracle.py")
    try:
        import psutil
    except ImportError:
        return []
    mine = {me_pid}
    try:
        mine |= {p.pid for p in psutil.Process(me_pid).parents()}
    except Exception:                                            # noqa: BLE001
        pass
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] in mine:
            continue
        cmd = " ".join(p.info.get("cmdline") or [])
        if any(s in cmd for s in pats) and "test_" not in cmd:
            out.append(f"pid {p.info['pid']}: {cmd[:140]}")
    return out


def idle_reasons(lane_me: str) -> list[str]:
    why = []
    lc = _longctx_run()
    rec = lc.other_pause(lane_me)
    if rec:
        why.append(f"gpu lane paused by {rec.get('by')!r}")
    why += [f"benchmark running: {b}" for b in busy_elsewhere(os.getpid())]
    xs = []
    for _ in range(10):
        g = gpu_read()
        if g:
            xs.append(g["util"])
        time.sleep(1)
    if xs and statistics.mean(xs) > 30:
        why.append(f"5060 Ti busy: {statistics.mean(xs):.0f}% mean utilisation over 10 s")
    if PROD_ID in running_ids():
        st, txt = http("GET", f"{SWAP}/upstream/{PROD_ID}/slots", timeout=20)
        try:
            busy = [s.get("id") for s in json.loads(txt) if s.get("is_processing")]
        except Exception:                                        # noqa: BLE001
            busy = []
        if busy:
            why.append(f"{PROD_ID} slots processing: {busy}")
    return why


def preflight(need_quality: bool) -> list[str]:
    """What must be true before the window is used. Empty list = go."""
    bad = []
    if not os.path.exists(BIAS) or os.path.getsize(BIAS) == 0:
        bad.append(f"bias file missing: {BIAS} (docs/CONTEXT-EXPANSION.md s.3 makes it)")
    p = profiles()
    if p is None:
        bad.append(f"llama-swap {SWAP}/api/profiles did not answer")
    else:
        have = {x.get("id") for x in p.get("profiles") or []}
        want = {prof for _, prof in CANDIDATES.values()}
        if not want <= have:
            bad.append("the running llama-swap does not know the context-trial profiles "
                       f"(has {sorted(have)}): it was started before config.yaml gained "
                       "them. Restart the stack inside the window first.")
        if p.get("active"):
            bad.append(f"a profile is already active: {p.get('active')!r}")
    st, _ = http("GET", f"{PROXY}/health", timeout=10)
    if st != 200:
        bad.append(f"proxy {PROXY}/health -> {st}")
    if need_quality and not shutil.which("wsl.exe"):
        bad.append("wsl.exe not found (quality phase runs LiveBench in WSL)")
    return bad


# ------------------------------------------------------------- plan --------
def plan(items: int, reps: int, cand: int = 262_144) -> dict:
    """Request counts and an ASSUMED-rate time estimate. Sends nothing."""
    rates = {"prefill_tps": 300.0, "decode_tps": 40.0, "answer_tokens": 60,
             "speed_tokens": 350, "note": "assumed, not measured on this card"}
    out = {"rates": rates, "arms": {}}
    fit_tokens = sum(int(c * 0.85) for c in CANDIDATES)
    out["fit"] = {"candidates": sorted(CANDIDATES, reverse=True),
                  "hours": round((fit_tokens / rates["prefill_tps"] + 2 * 120) / 3600, 2)}
    for arm, pool in (("q8", PROD_C), ("q4", cand)):
        ladder = [L for L in SHARED_LADDER + (BEYOND_LADDER.get(cand, []) if arm == "q4" else [])
                  if fits(L, pool)]
        acc_req = len(ladder) * items * len(hs.TASKS)
        acc_s = sum(L / rates["prefill_tps"] for L in ladder) * items \
            + acc_req * rates["answer_tokens"] / rates["decode_tps"]
        sp = [L for L in SPEED_LENGTHS if fits(L, pool)]
        sp_s = sum(L / rates["prefill_tps"] for L in sp) * reps \
            + len(sp) * reps * 2 * rates["speed_tokens"] / rates["decode_tps"]
        out["arms"][arm] = {"pool": pool, "ladder": ladder, "accuracy_requests": acc_req,
                            "speed_requests": len(sp) * reps * 2,
                            "hours_accuracy": round(acc_s / 3600, 1),
                            "hours_speed": round(sp_s / 3600, 1),
                            "hours_quality": 1.5}
    out["hours_total"] = round(out["fit"]["hours"] + sum(
        a["hours_accuracy"] + a["hours_speed"] + a["hours_quality"] for a in out["arms"].values()), 1)
    return out


# ------------------------------------------------------------- main --------
def main(argv: list[str]) -> int:
    global _EVENTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", action="store_true", help="print the plan; send nothing")
    ap.add_argument("--restore", action="store_true", help="put production back and exit")
    ap.add_argument("--window", action="store_true",
                    help="the operator grants a maintenance window (production is unloaded)")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--phases", default="fit,speed,accuracy,quality")
    ap.add_argument("--floor-mib", type=int, default=GATES["floor_mib"],
                    help="free-VRAM floor for FIT, fixed in the manifest at the first start")
    ap.add_argument("--items", type=int, default=12)
    ap.add_argument("--speed-reps", type=int, default=3)
    ap.add_argument("--wait-idle", type=float, default=0.0, help="minutes to wait for an idle card")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--wsl-distro", default=None)
    ap.add_argument("--lb-api-base", default=None, help="LiveBench's view of :1234 from WSL")
    a = ap.parse_args(argv)
    phases = [p for p in a.phases.split(",") if p]

    if a.plan:
        print(vram.render())
        print()
        for c in sorted(CANDIDATES, reverse=True):
            print(f"  plan if {c:,} passes FIT:")
            print(json.dumps(plan(a.items, a.speed_reps, c), indent=2))
        print(f"\n  bias file {'present' if os.path.exists(BIAS) else 'MISSING'}: {BIAS}")
        miss = livebench_arms_missing()
        print(f"  LiveBench arms {'present' if not miss else 'MISSING: ' + ', '.join(miss)}")
        print(f"  gates (pre-registered): {json.dumps(dict(GATES, floor_mib=a.floor_mib))}")
        return 0

    if a.restore:
        ensure_production()
        return 0

    if not a.window:
        print("  refusing: this run unloads the production model and reroutes `bonsai`.\n"
              "  Pass --window only inside a maintenance window the operator has given.")
        return 3
    if not a.key_file or not os.path.exists(a.key_file):
        print("  --key-file is required (the proxy needs a key; it is never printed)")
        return 2
    key = open(a.key_file, encoding="utf-8").read().strip()
    run_id = a.run_id or time.strftime("kvq4-%Y%m%d-%H%M%S")
    run_dir = os.path.join(RESULTS, run_id)
    os.makedirs(run_dir, exist_ok=True)
    _EVENTS = os.path.join(run_dir, "events.jsonl")

    bad = preflight("quality" in phases)
    if bad:
        for b in bad:
            log(f"PREFLIGHT: {b}", kind="preflight")
        return 3
    lane_me = f"bench/kv_context/run.py {run_id}"
    t_wait = time.time() + 60 * a.wait_idle
    while True:
        why = idle_reasons(lane_me)
        if not why:
            break
        for w in why:
            log(f"BUSY: {w}")
        if time.time() > t_wait:
            log("not starting: one GPU consumer at a time (AGENTS.md)")
            return 3
        time.sleep(60)

    man_path = os.path.join(run_dir, "manifest.json")
    man = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else {}
    if man.get("gates") and man["gates"].get("floor_mib") != a.floor_mib:
        log(f"refusing to resume: manifest floor {man['gates']['floor_mib']} != --floor-mib {a.floor_mib}")
        return 2
    man.setdefault("gates", dict(GATES, floor_mib=a.floor_mib))
    man.setdefault("started", time.time())
    chunks = synth_code.chunks(SYNTH_SEED)
    man.update(run_id=run_id, card=CARD_UUID, proxy=PROXY, swap=SWAP, bias=BIAS,
               candidates={str(c): list(v) for c, v in CANDIDATES.items()},
               synth={"seed": SYNTH_SEED, "fingerprint": synth_code.fingerprint(chunks)},
               item_seed=ITEM_SEED, items=a.items, speed_reps=a.speed_reps,
               features=FEATURES, shared_ladder=SHARED_LADDER, beyond_ladder=BEYOND_LADDER,
               projection=vram.table(), updated=time.time())

    lc = _longctx_run()
    lc.pause_lane(lane_me, "KV-quantisation context trial (docs/CONTEXT-EXPANSION.md)")
    def unload_trials():
        ids = running_ids()
        for mid, _ in CANDIDATES.values():
            if mid in ids:
                unload(mid)
    guard = VramGuard(on_abort=unload_trials)
    guard.start()
    stop_snap = threading.Event()

    def snapshots():
        while not stop_snap.is_set():
            try:
                log("snapshot", kind="snapshot", running=[
                    {"model": m.get("model"), "state": m.get("state")} for m in running()],
                    gpu=gpu_read())
                lc.pause_lane(lane_me, "KV-quantisation context trial")
            except Exception:                                    # noqa: BLE001
                pass
            stop_snap.wait(60)
    threading.Thread(target=snapshots, daemon=True).start()
    deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
    px = Proxy(key, run_id)
    stopped = None
    try:
        ensure_production()
        if not man.get("calibration"):
            man["calibration"] = calibrate(px, chunks)
        json.dump(man, open(man_path, "w", encoding="utf-8"), indent=2)
        log(f"calibration {man['calibration']}")

        if "fit" in phases:
            phase_fit(px, man, run_dir, guard, a.floor_mib, chunks)
            ensure_production()
        cand = adopted_candidate(run_dir)
        man["adopted_candidate"] = cand
        json.dump(man, open(man_path, "w", encoding="utf-8"), indent=2)
        if cand is None:
            stopped = "no candidate passed FIT; nothing further is measured on q4_0"
            log(stopped)
            return 0
        for arm, pool in (("q8", PROD_C), ("q4", cand)):
            arm_state(arm, cand)
            if "speed" in phases:
                stopped = phase_speed(px, man, run_dir, arm, pool, chunks, a.speed_reps, guard, deadline)
                if stopped:
                    break
            if "accuracy" in phases:
                ladder = SHARED_LADDER + (BEYOND_LADDER.get(cand, []) if arm == "q4" else [])
                stopped = phase_accuracy(px, man, run_dir, arm, pool, ladder, a.items, chunks,
                                         guard, deadline)
                if stopped:
                    break
            if "quality" in phases:
                msg = phase_quality(run_dir, arm, a.key_file, a.wsl_distro, a.lb_api_base)
                man.setdefault("quality", {})[arm] = msg or "ok"
                json.dump(man, open(man_path, "w", encoding="utf-8"), indent=2)
        return 1 if stopped and "outage" in stopped else 0
    finally:
        stop_snap.set()
        man["stopped"] = stopped
        man["updated"] = time.time()
        try:
            json.dump(man, open(man_path, "w", encoding="utf-8"), indent=2)
        except Exception:                                        # noqa: BLE001
            pass
        try:
            ensure_production()
        except Exception as e:                                   # noqa: BLE001
            log(f"RESTORE FAILED: {e}. Run: python bench/kv_context/run.py --restore", kind="restore")
        guard.stop()
        lc.resume_lane(lane_me)
        log(f"stopped: {stopped}" if stopped else "finished")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
