#!/usr/bin/env python
"""PERFORMANCE: speed per model, both GPUs over time, the GPU gates' arms and
the model swaps, for the dashboard's SOKUDO page (the retired benchmark
page's place).

Operator, 2026-09-30: "we can probably just fully retire the benchmark page,
or repurpose it for just a performance metrics page all around tok/s per
model graphs of speed, both GPUs etc. ... a deep dive page if enough
information is available."

    GET /dash/api/perf            the last 24 hours
    GET /dash/api/perf/<window>   1h | 6h | 24h | 7d | 30d
    GET /dash/api/perf/<window>/<traffic>   client (default) | test | all

`by_tier` (2026-10-06) is decode and prefill tok/s per effort tier and per
model, by context bucket, for the traffic asked for (tier_section); every
other section is as it was, over all traffic.

Under /dash/api (gated in server.py). READ-ONLY and MODEL-FREE: nothing here
asks llama-swap or a model server anything (mcp/test_dash_no_load.py).

  real traffic   index/stats.sqlite3 (mcp/stats_store.py, since
                 2026-09-30): every generation's model, role, prompt tokens
                 (reused / processed), prefill ms, completion tokens and
                 decode rate, from llama-server's own `timings`
  both GPUs      the same store's per-minute rows from the power sampler's
                 1 s nvidia-smi reads (utilisation, VRAM, watts,
                 temperature)
  swaps          max_mode.wait_ready's swaps (from, to, load seconds)
  gates          the GPU gates' own result files, each run as measured with
                 its n: the Flash-Next gate (~/octo/flashnext-gate-*/
                 gate.json), the Mirai S gate (~/octo/mirai-s-gate-*/
                 gate.json, when one has run), bench/results/kv_rank/*
                 (YAMADORI_GATE_GLOBS adds paths)

CONTEXT BINS are the depths the gates measure at (4K, 8K, 32K, 64K, 128K:
bench/flashnext_gate.py, bench/mirai_s_gate.py `speed`), so real traffic
sits beside the gate's arms at the same depth. A decode rate is taken as
llama-server reports it (predicted_n / predicted_ms); a one-token decider
read has none and is counted apart (role `decider`).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stats_store  # noqa: E402
from dash_jjava import WINDOWS, DEFAULT_WINDOW, bucket_of, pct, spread  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# the GPU history's bucket per window: the minute rows as they are for an
# hour, folded for the longer windows (one point per bucket per card)
GPU_BUCKET = {"1h": 60, "6h": 300, "24h": 900, "7d": 3600, "30d": 6 * 3600}
# The gates' measurement depths (the `speed` steps' labels 4k/32k/64k/128k
# and the lane's 8K), as bin edges in tokens.
CTX_EDGES = (4096, 8192, 32768, 65536, 131072)
SCATTER_MAX = 600
GENERATION_ROLES = ("main", "side_call", "internal", "second_brain")

# THE TIER TABLE (operator, 2026-10-06: "What are we averaging on tok/s per
# model, are we collecting those stats ... at least by 'effort' tier since we
# consider different qwen variants the same model behind our proxy").
# DISPLAY BUCKETS for the context size, in prompt tokens: they group what is
# shown and decide nothing (the gates' own depths stay CTX_EDGES above).
TIER_CTX = ((0, 8192, "0-8K"), (8192, 32768, "8-32K"),
            (32768, 65536, "32-64K"), (65536, None, "64K+"))
# A generation's prefill rate is read only where it processed this many
# tokens (the operator's figure, 2026-10-06): a short prompt's time is mostly
# fixed cost, not a rate.
PREFILL_MIN_PROCESSED = 2048
# traffic filter (stats_store `traffic`): the operator's own use, the live
# suites' and soaks' test accounts, or both. Rows with no traffic class (the
# worker's jobs, and every row before 2026-10-06) are in `all` only.
TRAFFIC = ("client", "test", "all")
DEFAULT_TRAFFIC = "client"
# conversation roles; a decider read is one token and has no rate, a warm is
# not a generation anyone waits for
TIER_ROLES = ("main", "side_call", "internal")
CACHE_S = 15
GATE_GROUPS_MAX = 400          # a memory guard per file

_cache: dict = {}
_gate_cache: dict = {}
_lock = threading.Lock()


def _json(code: int, payload: dict):
    return code, "application/json", json.dumps(payload, default=str).encode()


def ctx_bin(prompt: int | None) -> str | None:
    if not isinstance(prompt, (int, float)):
        return None
    lo = 0
    for e in CTX_EDGES:
        if prompt <= e:
            return f"{lo // 1024}K-{e // 1024}K"
        lo = e
    return f">{CTX_EDGES[-1] // 1024}K"


def bins_order() -> list[str]:
    out, lo = [], 0
    for e in CTX_EDGES:
        out.append(f"{lo // 1024}K-{e // 1024}K")
        lo = e
    return out + [f">{CTX_EDGES[-1] // 1024}K"]


def prompt_tps(r: dict) -> float | None:
    p, ms = r.get("processed"), r.get("prompt_ms")
    if isinstance(p, (int, float)) and isinstance(ms, (int, float)) and p > 0 and ms > 0:
        return round(p * 1000.0 / ms, 2)
    return None


# ------------------------------------------------------------ generations --
def models_section(rows: list[dict], since: float, step: int, n: int) -> list[dict]:
    per: dict[str, dict] = {}
    for r in rows:
        m = str(r.get("model") or "unknown")
        p = per.setdefault(m, {"model": m, "generations": 0, "roles": {},
                               "dec": [], "pre": [], "dec_b": [[] for _ in range(n)],
                               "pre_b": [[] for _ in range(n)], "n_b": [0] * n,
                               "ctx": {}, "scatter": [], "decider_pre": [],
                               "completion": 0, "processed": 0, "reused": 0})
        role = str(r.get("role") or "?")
        p["generations"] += 1
        p["roles"][role] = p["roles"].get(role, 0) + 1
        p["completion"] += int(r.get("completion") or 0)
        p["processed"] += int(r.get("processed") or 0)
        p["reused"] += int(r.get("reused") or 0)
        b = bucket_of(float(r["ts"]), since, step, n)
        pt = prompt_tps(r)
        if role == "decider":
            if pt is not None:
                p["decider_pre"].append(pt)
            continue
        d = r.get("decode_tps")
        dec = float(d) if isinstance(d, (int, float)) and d > 0 and \
            (r.get("completion") or 0) > 1 else None
        if b is not None:
            p["n_b"][b] += 1
        if dec is not None:
            p["dec"].append(dec)
            if b is not None:
                p["dec_b"][b].append(dec)
        if pt is not None:
            p["pre"].append(pt)
            if b is not None:
                p["pre_b"][b].append(pt)
        cb = ctx_bin(r.get("prompt"))
        if cb:
            c = p["ctx"].setdefault(cb, {"dec": [], "pre": [], "n": 0})
            c["n"] += 1
            if dec is not None:
                c["dec"].append(dec)
            if pt is not None:
                c["pre"].append(pt)
        if dec is not None and isinstance(r.get("prompt"), (int, float)):
            p["scatter"].append([int(r["prompt"]), round(dec, 2), role])
    out = []
    for p in sorted(per.values(), key=lambda x: -x["generations"]):
        out.append({
            "model": p["model"], "generations": p["generations"],
            "roles": p["roles"], "tokens": {"completion": p["completion"],
                                            "processed": p["processed"],
                                            "reused": p["reused"]},
            "decode_tps": spread(p["dec"]), "prompt_tps": spread(p["pre"]),
            "decider_read_prompt_tps": spread(p["decider_pre"]),
            "series": {"decode_p50": [pct(x, 0.5) for x in p["dec_b"]],
                       "decode_p10": [pct(x, 0.1) for x in p["dec_b"]],
                       "decode_p90": [pct(x, 0.9) for x in p["dec_b"]],
                       "prompt_p50": [pct(x, 0.5) for x in p["pre_b"]],
                       "n": p["n_b"]},
            "by_ctx": [{"bin": k, "n": p["ctx"][k]["n"],
                        "decode": spread(p["ctx"][k]["dec"]),
                        "prompt": spread(p["ctx"][k]["pre"])}
                       for k in bins_order() if k in p["ctx"]],
            "scatter": p["scatter"][-SCATTER_MAX:]})
    return out


# ------------------------------------------------------------- by tier ------
def tier_ctx(prompt) -> str | None:
    if not isinstance(prompt, (int, float)):
        return None
    for lo, hi, name in TIER_CTX:
        if hi is None or prompt <= hi:
            return name
    return None


def dist(xs: list[float]) -> dict:
    """n, p50, mean and p90 of a list of rates (nearest-rank percentiles, as
    everywhere on this page); n 0 and nulls when empty."""
    xs = [float(x) for x in xs if isinstance(x, (int, float))]
    return {"n": len(xs), "p50": pct(xs, 0.5),
            "mean": round(sum(xs) / len(xs), 2) if xs else None,
            "p90": pct(xs, 0.9)}


def _rates(r: dict) -> tuple[float | None, float | None, bool]:
    """(decode tok/s, prefill tok/s, cold) of one generation row: the decode
    rate as llama-server reported it, where it decoded more than one token
    (a one-token read has none); the prefill rate where it processed at least
    PREFILL_MIN_PROCESSED tokens; cold is the row's own flag (NULL = not
    known to be cold)."""
    d = r.get("decode_tps")
    dec = float(d) if isinstance(d, (int, float)) and d > 0 and \
        (r.get("completion") or 0) > 1 else None
    pre = prompt_tps(r) if (r.get("processed") or 0) >= PREFILL_MIN_PROCESSED \
        else None
    return dec, pre, r.get("cold") == 1


class _Acc:
    """One group's rates, whole and by context bucket."""

    def __init__(self) -> None:
        self.n = 0
        self.dec: list[float] = []
        self.warm: list[float] = []
        self.cold: list[float] = []
        self.ctx: dict[str, "_Acc"] = {}

    def add(self, r: dict, by_ctx: bool = True) -> None:
        dec, pre, cold = _rates(r)
        self.n += 1
        if dec is not None:
            self.dec.append(dec)
        if pre is not None:
            (self.cold if cold else self.warm).append(pre)
        if by_ctx:
            b = tier_ctx(r.get("prompt"))
            if b:
                self.ctx.setdefault(b, _Acc()).add(r, by_ctx=False)

    def out(self) -> dict:
        d = {"n": self.n, "decode": dist(self.dec),
             "prefill_warm": dist(self.warm), "prefill_cold": dist(self.cold)}
        if self.ctx:
            d["by_ctx"] = [{"bucket": name, **self.ctx[name].out()}
                           for _, _, name in TIER_CTX if name in self.ctx]
        return d


def tier_section(rows: list[dict], traffic: str = DEFAULT_TRAFFIC,
                 window: str | None = None) -> dict:
    """Decode and prefill tok/s per effort TIER and per model, by context
    bucket, from the generation rows; `traffic` picks which rows (TRAFFIC).
    Rows before 2026-10-06 have no tier and no traffic class: they are the
    tier `null` and only in `all`. Every number is from the rows."""
    if traffic not in TRAFFIC:
        traffic = DEFAULT_TRAFFIC
    per: dict[tuple, _Acc] = {}
    per_model: dict[tuple, _Acc] = {}
    seen = {"client": 0, "test": 0, "unrecorded": 0}
    left_out = {"traffic": 0, "role": 0}
    for r in rows:
        t = r.get("traffic")
        seen[t if t in ("client", "test") else "unrecorded"] += 1
        role = str(r.get("role") or "?")
        if role not in TIER_ROLES:
            left_out["role"] += 1
            continue
        if traffic != "all" and t != traffic:
            left_out["traffic"] += 1
            continue
        model = str(r.get("model") or "unknown")
        per.setdefault((r.get("tier") or None, model, role), _Acc()).add(r)
        per_model.setdefault((model, role), _Acc()).add(r)

    def order(k):                     # recorded tiers first, then the unrecorded
        return (k[0] is None, k[0] or "", k[1], k[2])
    return {
        "traffic": traffic, "traffic_names": list(TRAFFIC),
        "window": window, "ctx_buckets": [n for _, _, n in TIER_CTX],
        "prefill_min_processed": PREFILL_MIN_PROCESSED,
        "generations": {"in_window": len(rows), **seen},
        "left_out": left_out,
        "rows": [{"tier": k[0], "model": k[1], "role": k[2], **per[k].out()}
                 for k in sorted(per, key=order)],
        "by_model": [{"model": k[0], "role": k[1], **per_model[k].out()}
                     for k in sorted(per_model)],
        "rules": ("decode: llama-server's predicted_n / predicted_ms where "
                  "more than one token was decoded; prefill: processed * "
                  f"1000 / prompt_ms where >= {PREFILL_MIN_PROCESSED} tokens "
                  "were processed, `cold` where the request loaded or read "
                  "in its model (stats_store `cold`), else warm; context "
                  "buckets by the prompt's tokens; tier null = recorded "
                  "before the tier column existed; roles main, side_call and "
                  "internal"),
    }


# -------------------------------------------------------------------- GPUs --
def gpus_section(since: float, window: str) -> dict:
    step = GPU_BUCKET.get(window, 900)
    rows = stats_store.read("gpu", since)
    cards: dict[int, dict] = {}
    for r in rows:
        idx = int(r.get("idx") or 0)
        c = cards.setdefault(idx, {"idx": idx, "name": r.get("name"),
                                   "uuid": r.get("uuid"), "total": r.get("total"),
                                   "b": {}})
        c["total"] = r.get("total") or c["total"]
        k = int(float(r["ts"]) // step) * step
        a = c["b"].setdefault(k, {"n": 0, "util": 0.0, "util_n": 0,
                                  "util_max": None, "used_max": None,
                                  "watts": 0.0, "watts_n": 0, "temp_max": None})
        a["n"] += int(r.get("n") or 0)
        if isinstance(r.get("util_avg"), (int, float)):
            a["util"] += float(r["util_avg"])
            a["util_n"] += 1
        for key, src in (("util_max", "util_max"), ("used_max", "used_max"),
                         ("temp_max", "temp_max")):
            v = r.get(src)
            if isinstance(v, (int, float)):
                a[key] = v if a[key] is None else max(a[key], v)
        if isinstance(r.get("watts_avg"), (int, float)):
            a["watts"] += float(r["watts_avg"])
            a["watts_n"] += 1
    out = []
    for idx in sorted(cards):
        c = cards[idx]
        keys = sorted(c["b"])
        ser = {"t": keys, "util_avg": [], "util_max": [], "used_max": [],
               "watts_avg": [], "temp_max": [], "samples": []}
        for k in keys:
            a = c["b"][k]
            ser["util_avg"].append(round(a["util"] / a["util_n"], 1) if a["util_n"] else None)
            ser["util_max"].append(a["util_max"])
            ser["used_max"].append(a["used_max"])
            ser["watts_avg"].append(round(a["watts"] / a["watts_n"], 1) if a["watts_n"] else None)
            ser["temp_max"].append(a["temp_max"])
            ser["samples"].append(a["n"])
        out.append({"idx": idx, "name": c["name"], "uuid": c["uuid"],
                    "total_mib": c["total"], "series": ser})
    main_uuid = os.environ.get("YAMADORI_MAIN_GPU_UUID",
                               "GPU-de660e90-0e9c-d465-b389-6df63021b920")
    for c in out:
        c["main"] = c.get("uuid") == main_uuid
    return {"bucket_s": step, "cards": out, "rows": len(rows)}


# ------------------------------------------------------------------- swaps --
def swaps_section(since: float) -> dict:
    rows = []
    for r in stats_store.read("swaps", since):
        try:
            frm = json.loads(r.get("from_models") or "[]")
            left = json.loads(r.get("left_loaded") or "[]")
        except ValueError:
            frm, left = [], []
        rows.append({"ts": r["ts"], "from": frm, "to": r.get("to_model"),
                     "load_s": r.get("load_s"), "ok": bool(r.get("ok")),
                     "how": r.get("how"), "left_loaded": left})
    by_to: dict[str, list[float]] = {}
    for r in rows:
        if isinstance(r["load_s"], (int, float)):
            by_to.setdefault(str(r["to"]), []).append(float(r["load_s"]))
    last = None
    try:
        import max_mode
        snap = max_mode.snapshot()
        last = snap.get("last_swap") or None
        table = {"enabled": snap.get("enabled"), "tiers": snap.get("tiers"),
                 "models": snap.get("models"), "main": snap.get("main")}
    except Exception as e:                                       # noqa: BLE001
        table = {"error": f"{type(e).__name__}: {e}"[:200]}
    return {"rows": rows[::-1], "load_s": {k: spread(v) for k, v in by_to.items()},
            "last_in_memory": last, "table": table}


# ------------------------------------------------------------------- gates --
def gate_globs() -> list[tuple[str, str]]:
    """(kind, glob) of every result file read."""
    # the gates' --out folders (bench/flashnext_gate.py, bench/mirai_s_gate.py
    # write gate.json there); YAMADORI_OCTO_DIR moves it (the tests)
    octo = os.environ.get("YAMADORI_OCTO_DIR") or os.path.join(
        os.path.expanduser("~"), "octo")
    out = [("flash-next gate", os.path.join(octo, "flashnext-gate-*", "gate.json")),
           ("mirai-s gate", os.path.join(octo, "mirai*gate*", "gate.json")),
           ("kv_rank", os.path.join(ROOT, "bench", "results", "kv_rank", "*", "measure", "*.json")),
           ("kv_rank", os.path.join(ROOT, "bench", "results", "kv_rank", "*", "line", "line-*.json"))]
    for g in (os.environ.get("YAMADORI_GATE_GLOBS") or "").split(os.pathsep):
        if g.strip():
            out.append(("other", g.strip()))
    return out


def _is_run(d) -> bool:
    return isinstance(d, dict) and (
        ("prompt_tps" in d or "prompt_per_second" in d) and
        ("tps" in d or "predicted_per_second" in d))


def _run_rates(d: dict) -> tuple[float | None, float | None, int | None, int | None]:
    dec = d.get("tps", d.get("predicted_per_second"))
    pre = d.get("prompt_tps", d.get("prompt_per_second"))
    pn = d.get("predicted_n")
    dec = float(dec) if isinstance(dec, (int, float)) and dec > 0 and \
        (pn is None or pn > 1) else None
    pre = float(pre) if isinstance(pre, (int, float)) and pre > 0 else None
    return dec, pre, d.get("prompt_n"), d.get("n_ctx_used")


def _med(xs: list[float]) -> float | None:
    return pct(xs, 0.5)


def walk_runs(obj, path: tuple = ()) -> list[tuple[tuple, list[dict]]]:
    """Every list of measured runs in a result file, with its key path."""
    found: list[tuple[tuple, list[dict]]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("cmd", "_log", "env", "arg_overrides"):
                continue
            if _is_run(v):
                found.append((path + (k,), [v]))
            else:
                found += walk_runs(v, path + (k,))
    elif isinstance(obj, list):
        runs = [x for x in obj if _is_run(x)]
        if runs:
            found.append((path, runs))
        else:
            for i, v in enumerate(obj[:50]):
                if isinstance(v, (dict, list)):
                    found += walk_runs(v, path + (str(i),))
    return found


def _passes(obj, path: tuple = ()) -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "PASS" and isinstance(v, bool):
                out["/".join(path) or "gate"] = v
            elif isinstance(v, dict):
                out.update(_passes(v, path + (k,)))
    return out


def parse_gate(kind: str, path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    groups = []
    for kp, runs in walk_runs(g)[:GATE_GROUPS_MAX]:
        if kp and kp[0] == "arms" and len(kp) > 1:
            arm, what = kp[1], "/".join(kp[2:])
        else:
            arm = (g.get("arm") if isinstance(g, dict) else None) or \
                os.path.basename(os.path.dirname(os.path.dirname(path)))
            what = "/".join(kp)
        decs, pres, pns, ctxs = [], [], [], []
        for r in runs:
            d, p, pn, cu = _run_rates(r)
            if d is not None:
                decs.append(d)
            if p is not None:
                pres.append(p)
            if isinstance(pn, (int, float)):
                pns.append(pn)
            if isinstance(cu, (int, float)):
                ctxs.append(cu)
        groups.append({"arm": arm, "what": what, "n": len(runs),
                       "decode": {"median": _med(decs), "min": min(decs) if decs else None,
                                  "max": max(decs) if decs else None, "n": len(decs)},
                       "prompt": {"median": _med(pres), "min": min(pres) if pres else None,
                                  "max": max(pres) if pres else None, "n": len(pres)},
                       "prompt_n": _med(pns), "ctx_used": _med(ctxs)})
    st = os.stat(path)
    return {"kind": kind, "source": os.path.basename(os.path.dirname(path))
            if os.path.basename(path) == "gate.json" else
            os.path.relpath(path, ROOT).replace("\\", "/"),
            "file": os.path.basename(path),
            "started": g.get("started") if isinstance(g, dict) else None,
            "finished": g.get("finished") if isinstance(g, dict) else None,
            "steps": g.get("steps") if isinstance(g, dict) else None,
            "arms": sorted((g.get("arms") or {}).keys()) if isinstance(g, dict)
            and isinstance(g.get("arms"), dict) else None,
            "pass": _passes(g), "groups": groups, "mtime": st.st_mtime,
            "in_progress": isinstance(g, dict) and "started" in g
            and "finished" not in g and kind.endswith("gate")}


def gates_section() -> dict:
    files, errors = [], []
    for kind, pattern in gate_globs():
        for p in sorted(glob.glob(pattern)):
            try:
                st = os.stat(p)
                key = (p, st.st_mtime_ns, st.st_size)
                with _lock:
                    hit = _gate_cache.get(p)
                if hit and hit[0] == key:
                    files.append(hit[1])
                    continue
                d = parse_gate(kind, p)
                with _lock:
                    _gate_cache[p] = (key, d)
                files.append(d)
            except Exception as e:                               # noqa: BLE001
                errors.append({"file": os.path.basename(p),
                               "error": f"{type(e).__name__}: {e}"[:200]})
    files.sort(key=lambda d: d.get("mtime") or 0, reverse=True)
    return {"files": files, "errors": errors,
            "globs": [{"kind": k, "glob": g.replace(os.path.expanduser("~"), "~")}
                      for k, g in gate_globs()]}


LEFT_OUT = [
    {"graph": "tok/s before this deploy",
     "why": "no generation's speed was kept anywhere (x_yamadori.cache went to "
            "the client only; the token ledger keeps daily token counts, not "
            "rates): the history starts when mcp/stats_store.py ships"},
    {"graph": "per-model VRAM",
     "why": "nvidia-smi reports memory per card, not per llama-server; the "
            "card history is per card, and the model on the card is the swap "
            "history's"},
    {"graph": "swap load times from llama-swap's own log",
     "why": "logs/stack.log (llama-swap) carries no timestamps; swaps are the "
            "ones max_mode.wait_ready makes (the proxy), since this deploy"},
    {"graph": "GPU clocks, fan, PCIe",
     "why": "not sampled: the power sampler's nvidia-smi query reads "
            "utilisation, memory, power and (since this deploy) temperature"},
    {"graph": "Mirai S gate",
     "why": "shown when ~/octo/mirai*gate*/gate.json exists; bench/"
            "mirai_s_gate.py has not written one yet"},
]


# -------------------------------------------------------------------- build --
def overview(window: str = DEFAULT_WINDOW, now: float | None = None,
             traffic: str = DEFAULT_TRAFFIC) -> dict:
    now = time.time() if now is None else now
    span, step = WINDOWS[window]
    n = int(span // step)
    since = (int(now // step) + 1) * step - n * step
    out: dict = {"window": {"name": window, "since": since, "until": now,
                            "bucket_s": step,
                            "buckets": [since + i * step for i in range(n)],
                            "names": list(WINDOWS)},
                 "at": now, "ctx_bins": bins_order(),
                 "left_out": LEFT_OUT}
    gens = stats_store.read("generations", since)
    out["sources"] = {"stats": {k: v for k, v in stats_store.status().items()
                                if k != "db"},
                      "generations": len(gens)}
    for name, fn in (("models", lambda: models_section(gens, since, step, n)),
                     ("by_tier", lambda: tier_section(gens, traffic, window)),
                     ("gpus", lambda: gpus_section(since, window)),
                     ("swaps", lambda: swaps_section(since)),
                     ("gates", gates_section)):
        try:
            out[name] = fn()
        except Exception as e:                                   # noqa: BLE001
            out[name] = {"error": f"{name}: {type(e).__name__}: {e}"[:300]}
    return out


def cached(window: str, traffic: str = DEFAULT_TRAFFIC) -> dict:
    now = time.time()
    key = (window, traffic)
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_S:
            return hit[1]
    d = overview(window, now, traffic)
    with _lock:
        _cache[key] = (now, d)
    return d


def handle_get(path: str):
    """/dash/api/perf[/<window>[/<traffic>]], traffic one of TRAFFIC (default
    client). A `?traffic=` on the path is read too (server.dash_get passes
    the path without its query, so the path form is the contract)."""
    p, _, query = path.partition("?")
    p = p.rstrip("/")
    traffic = DEFAULT_TRAFFIC
    for kv in query.split("&"):
        k, _, v = kv.partition("=")
        if k == "traffic" and v:
            traffic = v
    if p == "/dash/api/perf":
        w = DEFAULT_WINDOW
    elif p.startswith("/dash/api/perf/"):
        parts = p[len("/dash/api/perf/"):].split("/")
        w = parts[0]
        if w not in WINDOWS:
            return _json(404, {"error": f"no window {w!r}: one of "
                                        f"{', '.join(WINDOWS)}"})
        if len(parts) > 2:
            return None
        if len(parts) == 2:
            traffic = parts[1]
    else:
        return None
    if traffic not in TRAFFIC:
        return _json(404, {"error": f"no traffic class {traffic!r}: one of "
                                    f"{', '.join(TRAFFIC)}"})
    try:
        return _json(200, cached(w, traffic))
    except Exception as e:                                       # noqa: BLE001
        return _json(500, {"error": f"perf overview raised "
                                    f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    d = overview(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WINDOW)
    print(json.dumps({"sources": d["sources"],
                      "gates": [(g["source"], len(g["groups"]))
                                for g in d["gates"].get("files", [])]},
                     indent=1, default=str))
