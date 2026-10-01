#!/usr/bin/env python
"""JJAVA: the decider's use, speed, answers and endpoints, for the dashboard
(the JJAVA page and the Skills page's injector panel).

Operator, 2026-09-30: "I also need jev/java stats, jev/java endpoint
visibility, graphs of our stats on it, when it is being used, etc. We should
have those stats built into the skills page too."

    GET /dash/api/jjava            the last 24 hours
    GET /dash/api/jjava/<window>   1h | 6h | 24h | 7d | 30d

Under /dash/api, which server.py gates with accounts.identify. READ-ONLY and
MODEL-FREE: every number comes from a record already written --

  logs/decider_decisions.jsonl   every decision the proxy's decider made
                                 (decide_turn DECISIONS, log v2): question
                                 set, model, the answer's distribution,
                                 confidence / noul, tier; since 2026-09-30
                                 also its ms and reads (latency)
  index/corpus.sqlite3 jev_call  every Jev API call (mcp/jev_api.py): route,
                                 status, model asked for and served, usage,
                                 ms, traffic class; 401 / 422-before-parse
                                 and GET /jev/v1/models since 2026-09-30
  index/stats.sqlite3            mcp/stats_store.py (2026-09-30): each
                                 request's decider Turn and injector record,
                                 and every slot release ("lane burst ended")
  bench/decider/results/models   each model's measured record (priors:
                                 measured or not; decider_bonsai.
                                 profile_status)

-- and nothing here reaches llama-swap's /upstream or a model server
(mcp/test_dash_no_load.py holds it to that). The decision log is read from
its tail, at most LOG_TAIL_BYTES (a memory guard; `sources` says when it
cut), and the answer is cached CACHE_S per window.
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stats_store  # noqa: E402

# name -> (seconds, bucket seconds)
WINDOWS = {"1h": (3600, 300), "6h": (6 * 3600, 900), "24h": (86400, 3600),
           "7d": (7 * 86400, 6 * 3600), "30d": (30 * 86400, 86400)}
DEFAULT_WINDOW = "24h"
LOG_TAIL_BYTES = 64 * 1024 * 1024
CACHE_S = 15
HIST_BINS = 10

JEV_ROUTES = (("POST", "/jev/v1/systemone"), ("POST", "/v1/systemone"),
              ("GET", "/jev/v1/models"))

# A question set's caller, by the family of its name (decide_turn MODES).
CALLER_OF = {"skill_item": "skills injector", "skill_inject": "skills injector",
             "choose": "skill selection", "pick": "skill selection",
             "package": "skill selection", "build_intent": "turn facts",
             "phase": "turn facts", "reads_package": "turn facts",
             "scratch_write": "turn facts", "verify": "verify"}

_cache: dict = {}
_cache_lock = threading.Lock()


def _json(code: int, payload: dict):
    return code, "application/json", json.dumps(payload, default=str).encode()


# ------------------------------------------------------------------ helpers --
def pct(xs: list[float], q: float) -> float | None:
    """Nearest-rank percentile of `xs` (q in [0, 1]); None when empty."""
    v = sorted(x for x in xs if isinstance(x, (int, float)))
    if not v:
        return None
    k = max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))
    return round(float(v[k]), 2)


def spread(xs: list[float]) -> dict:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return {"n": len(xs), "p50": pct(xs, 0.5), "p90": pct(xs, 0.9),
            "max": round(max(xs), 2) if xs else None}


def hist(xs: list[float], bins: int = HIST_BINS) -> list[int]:
    """Counts of values in [0, 1] over `bins` equal-width bins (1.0 in the
    last) -- JevBench's ECE binning (docs/JJAVA.md 3)."""
    out = [0] * bins
    for x in xs:
        if isinstance(x, (int, float)):
            out[min(bins - 1, max(0, int(float(x) * bins)))] += 1
    return out


def family(name: str) -> str:
    return str(name or "").split(":", 1)[0]


def question_set(name: str) -> str:
    """The set a decision is counted under: choose:stop is the stop judge;
    other choose:* rounds are one set; package:<p> and skill_item:<key> are
    their families."""
    fam = family(name)
    if name == "choose:stop":
        return "choose:stop"
    return fam


def caller_of(name: str) -> str:
    if name == "choose:stop":
        return "stop judge"
    return CALLER_OF.get(family(name), "other")


def answer_label(row: dict) -> str | None:
    """What the decision picked, as a label a person reads: a noul's
    true/false, a phase, a stop judge's option, "a skill" / "none" for a
    skill choice (its ids are hashes), a score level."""
    pick = row.get("pick") if row.get("pick") is not None else row.get("argmax")
    if pick is None:
        return "no pick (tie or low tier)"
    q = row.get("question") or {}
    ids = row.get("option_ids")
    keys = q.get("keys") or []
    if isinstance(ids, list) and pick in keys and len(ids) == len(keys):
        oid = ids[keys.index(pick)]
        if row.get("choose_kind") == "stop" or q.get("name") == "choose:stop":
            return str(oid)
        return "none" if oid == "none" else "an option"
    return str(pick)


def bucket_of(ts: float, since: float, step: int, n: int) -> int | None:
    i = int((ts - since) // step)
    return i if 0 <= i < n else None


# ------------------------------------------------------------------ sources --
def decisions_path() -> str:
    try:
        import decide_turn
        return decide_turn.DECISIONS
    except Exception:                                            # noqa: BLE001
        return os.environ.get("YAMADORI_DECIDER_DECISIONS") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "decider_decisions.jsonl")


def read_decisions(since: float, path: str | None = None) -> tuple[list[dict], dict]:
    """The v2 "decision" rows logged at or after `since`, oldest first, and
    what was read."""
    path = path or decisions_path()
    meta = {"path": os.path.basename(path), "exists": os.path.exists(path),
            "bytes_read": 0, "cut": False, "rows": 0, "bad_lines": 0,
            "legacy_rows": 0, "with_latency": 0}
    if not meta["exists"]:
        return [], meta
    rows: list[dict] = []
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > LOG_TAIL_BYTES:
                f.seek(size - LOG_TAIL_BYTES)
                f.readline()                     # a partial first line
                meta["cut"] = True
            data = f.read()
        meta["bytes_read"] = len(data)
    except OSError as e:
        meta["error"] = f"{type(e).__name__}: {e}"[:200]
        return [], meta
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            meta["bad_lines"] += 1
            continue
        if not isinstance(r, dict) or r.get("row") != "decision":
            continue
        if float(r.get("ts") or 0) < since:
            continue
        if r.get("v") != 2:
            meta["legacy_rows"] += 1
            continue
        if r.get("ms") is not None:
            meta["with_latency"] += 1
        rows.append(r)
    meta["rows"] = len(rows)
    rows.sort(key=lambda r: float(r.get("ts") or 0))
    return rows, meta


def corpus_path() -> str:
    try:
        import corpus
        return corpus.CORPUS_DB
    except Exception:                                            # noqa: BLE001
        return os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "index", "corpus.sqlite3")


def read_jev_calls(since: float, path: str | None = None) -> tuple[list[dict], dict]:
    """The corpus's `jev_call` events at or after `since`, read-only."""
    path = path or corpus_path()
    meta = {"path": os.path.basename(path), "exists": os.path.exists(path),
            "rows": 0}
    if not meta["exists"]:
        return [], meta
    out: list[dict] = []
    try:
        uri = pathlib.Path(path).resolve().as_uri() + "?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=2)
        try:
            for ts, payload in con.execute(
                    "SELECT ts, payload FROM events WHERE kind='jev_call' "
                    "AND ts >= ? ORDER BY ts", (since,)):
                try:
                    rec = json.loads(payload or "{}")
                except ValueError:
                    continue
                rec["ts"] = ts
                out.append(rec)
        finally:
            con.close()
    except sqlite3.Error as e:
        meta["error"] = f"{type(e).__name__}: {e}"[:200]
    meta["rows"] = len(out)
    return out, meta


def read_requests(since: float) -> list[dict]:
    out = []
    for r in stats_store.read("requests", since):
        try:
            r["rec"] = json.loads(r.get("rec") or "{}")
        except ValueError:
            r["rec"] = {}
        out.append(r)
    return out


# ---------------------------------------------------------------- sections --
def _series(n: int) -> list[int]:
    return [0] * n


def usage_section(rows: list[dict], calls: list[dict], since: float,
                  step: int, n: int) -> dict:
    by_set: dict[str, list[int]] = {}
    by_model: dict[str, list[int]] = {}
    by_caller: dict[str, list[int]] = {}
    totals: dict[str, int] = {}

    def add(d: dict, k: str, b: int | None, count: int = 1) -> None:
        if b is None:
            return
        d.setdefault(k, _series(n))[b] += count

    for r in rows:
        name = (r.get("question") or {}).get("name") or ""
        b = bucket_of(float(r["ts"]), since, step, n)
        qs = question_set(name)
        add(by_set, qs, b)
        add(by_model, str(r.get("model") or "?"), b)
        add(by_caller, caller_of(name), b)
        totals[qs] = totals.get(qs, 0) + 1
    for c in calls:
        qn = len(c.get("questions") or {})
        if not qn or int(c.get("status") or 0) != 200:
            continue
        b = bucket_of(float(c["ts"]), since, step, n)
        add(by_set, "jev api", b, qn)
        add(by_model, str(c.get("model") or "?"), b, qn)
        add(by_caller, f"jev {c.get('route') or '?'}", b, qn)
        totals["jev api"] = totals.get("jev api", 0) + qn
    return {"by_question_set": by_set, "by_model": by_model,
            "by_caller": by_caller, "totals": totals,
            "decisions": sum(totals.values())}


def latency_section(rows: list[dict], calls: list[dict], reqs: list[dict],
                    since: float, step: int, n: int) -> dict:
    per_read: list[float] = []
    per_decision: list[float] = []
    read_by_set: dict[str, list[float]] = {}
    read_by_model: dict[str, list[float]] = {}
    bursts: dict[tuple, float] = {}
    series_ms: list[list[float]] = [[] for _ in range(n)]
    for r in rows:
        ms = r.get("ms")
        if not isinstance(ms, (int, float)):
            continue
        reads = r.get("reads") or len(r.get("orders") or []) or 1
        pr = float(ms) / max(int(reads), 1)
        name = (r.get("question") or {}).get("name") or ""
        per_read.append(pr)
        per_decision.append(float(ms))
        read_by_set.setdefault(question_set(name), []).append(pr)
        read_by_model.setdefault(str(r.get("model") or "?"), []).append(pr)
        key = (r.get("request") or r.get("id"), r.get("kind"))
        bursts[key] = bursts.get(key, 0.0) + float(ms)
        b = bucket_of(float(r["ts"]), since, step, n)
        if b is not None:
            series_ms[b].append(float(ms))
    turn_ms = [float((r["rec"].get("decider") or {}).get("ms_questions"))
               for r in reqs if isinstance((r["rec"].get("decider") or {})
                                           .get("ms_questions"), (int, float))
               and (r["rec"].get("decider") or {}).get("decisions")]
    jev_ms, jev_read = [], []
    for c in calls:
        if int(c.get("status") or 0) != 200 or not c.get("questions") \
                or not isinstance(c.get("ms"), (int, float)):
            continue
        jev_ms.append(float(c["ms"]))
        reads = int(((c.get("usage") or {}).get("output_tokens")) or 0)
        if reads:
            jev_read.append(float(c["ms"]) / reads)
    return {
        "per_read": dict(spread(per_read), by_question_set={
            k: spread(v) for k, v in sorted(read_by_set.items())},
            by_model={k: spread(v) for k, v in sorted(read_by_model.items())}),
        "per_decision": spread(per_decision),
        "per_burst": {"decision_log": spread(list(bursts.values())),
                      "decider_turns": spread(turn_ms),
                      "jev_calls": spread(jev_ms)},
        "jev_per_read": spread(jev_read),
        "series": {"p50": [pct(x, 0.5) for x in series_ms],
                   "p90": [pct(x, 0.9) for x in series_ms],
                   "n": [len(x) for x in series_ms]},
        "note": ("a READ is one forward pass of one option order (a question "
                 "is read in two orders); a BURST is one request's decisions "
                 "of one kind (the decision log grouped by request), one "
                 "decider Turn's questions (x_yamadori), or one Jev call. "
                 "Decision rows carry ms since 2026-09-30; older rows count "
                 "in use, not in latency."),
    }


def question_sets_section(rows: list[dict]) -> list[dict]:
    sets: dict[str, dict] = {}
    for r in rows:
        q = r.get("question") or {}
        name = q.get("name") or ""
        qs = question_set(name)
        s = sets.setdefault(qs, {"name": qs, "type": q.get("type"),
                                 "caller": caller_of(name), "n": 0,
                                 "picks": {}, "tiers": {}, "models": {},
                                 "ties": 0, "orders_disagree": 0,
                                 "conf": [], "noul": []})
        s["n"] += 1
        lab = answer_label(r)
        s["picks"][lab] = s["picks"].get(lab, 0) + 1
        t = str(r.get("tier") or "untuned")
        s["tiers"][t] = s["tiers"].get(t, 0) + 1
        m = str(r.get("model") or "?")
        s["models"][m] = s["models"].get(m, 0) + 1
        if r.get("tie"):
            s["ties"] += 1
        if r.get("argmax_agree") is False:
            s["orders_disagree"] += 1
        if isinstance(r.get("confidence"), (int, float)):
            s["conf"].append(float(r["confidence"]))
        if isinstance(r.get("noul"), (int, float)):
            s["noul"].append(float(r["noul"]))
    out = []
    for s in sorted(sets.values(), key=lambda x: -x["n"]):
        conf, noul = s.pop("conf"), s.pop("noul")
        s["confidence_hist"] = hist(conf) if conf else None
        s["confidence"] = spread(conf) if conf else None
        s["noul_hist"] = hist(noul) if noul else None
        s["noul_middle"] = (sum(1 for x in noul if 0.35 <= x <= 0.65)
                            if noul else None)
        s["picks"] = dict(sorted(s["picks"].items(), key=lambda kv: -kv[1])[:12])
        out.append(s)
    return out


def thresholds_section(rows: list[dict]) -> dict:
    try:
        import decide_turn
        dt = dict(decide_turn.THRESHOLDS)
    except Exception as e:                                       # noqa: BLE001
        dt = {"error": f"{type(e).__name__}: {e}"[:200]}
    try:
        import skill_inject
        si = dict(skill_inject.THRESHOLDS)
    except Exception as e:                                       # noqa: BLE001
        si = {"error": f"{type(e).__name__}: {e}"[:200]}
    fired: dict[str, dict[str, int]] = {}
    for r in rows:
        qs = question_set((r.get("question") or {}).get("name") or "")
        t = str(r.get("tier") or "untuned")
        fired.setdefault(qs, {})
        fired[qs][t] = fired[qs].get(t, 0) + 1
    return {"decide_turn": dt, "skill_inject": si,
            "state": "untuned" if not dt and not si else "tuned rows present",
            "fired": fired,
            "rule": ("three tiers per question set and model (docs/JJAVA.md "
                     "3): high acts, medium proceeds with caution, low falls "
                     "back; an untuned set acts on the argmax. None ship "
                     "untuned (operator): the tables stay empty until an "
                     "owner pastes a tuned row.")}


def lane_section(since: float, step: int, n: int) -> dict:
    rows = stats_store.read("releases", since)
    by_why: dict[str, int] = {}
    burst = _series(n)
    kept = _series(n)
    for r in rows:
        why = str(r.get("why") or "?")
        by_why[why] = by_why.get(why, 0) + 1
        b = bucket_of(float(r["ts"]), since, step, n)
        if b is None:
            continue
        if why == "lane burst ended":
            burst[b] += 1
        if (r.get("skipped") or "").startswith("the lane is kept"):
            kept[b] += 1
    recent_mem = None
    try:
        import slots
        snap = slots.snapshot()
        recent_mem = (snap.get("release") or {}).get("recent")
        lane = snap.get("lane")
    except Exception:                                            # noqa: BLE001
        lane = None
    return {"releases": len(rows), "by_why": by_why,
            "lane_burst_ended": burst, "lane_kept": kept,
            "recent": [{k: r.get(k) for k in ("ts", "slot", "why", "by_why",
                                              "released", "skipped",
                                              "cells_before", "ms", "method")}
                       for r in rows[-40:]][::-1],
            "recent_in_memory": recent_mem, "lane": lane,
            "note": ("\"lane burst ended\" is a decider burst's release of "
                     "the lane (slots RELEASE); with layout v2 the lane is "
                     "KEPT and each burst leaves a \"kept\" note instead. "
                     "Recorded since 2026-09-30 (mcp/stats_store.py).")}


def jev_section(calls: list[dict], since: float, step: int, n: int) -> dict:
    routes = []
    for method, route in JEV_ROUTES:
        mine = [c for c in calls if c.get("route") == route]
        st: dict[str, int] = {}
        ser = _series(n)
        for c in mine:
            k = str(c.get("status") or "?")
            st[k] = st.get(k, 0) + 1
            b = bucket_of(float(c["ts"]), since, step, n)
            if b is not None:
                ser[b] += 1
        routes.append({"method": method, "route": route, "requests": len(mine),
                       "statuses": st, "series": ser})
    other = sorted({str(c.get("route")) for c in calls} -
                   {r for _, r in JEV_ROUTES})
    requested: dict[str, int] = {}
    served: dict[str, int] = {}
    traffic: dict[str, int] = {}
    errors: dict[str, int] = {}
    inp = outp = 0
    tok_series = _series(n)
    for c in calls:
        if c.get("requested"):
            requested[str(c["requested"])] = requested.get(str(c["requested"]), 0) + 1
        if c.get("model") and int(c.get("status") or 0) == 200:
            served[str(c["model"])] = served.get(str(c["model"]), 0) + 1
        tr = str(c.get("traffic") or "unknown")
        traffic[tr] = traffic.get(tr, 0) + 1
        if isinstance(c.get("error"), dict):
            k = f"{c['error'].get('status')} {c['error'].get('code') or ''}".strip()
            errors[k] = errors.get(k, 0) + 1
        u = c.get("usage") or {}
        i, o = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        inp += i
        outp += o
        b = bucket_of(float(c["ts"]), since, step, n)
        if b is not None:
            tok_series[b] += i + o
    cfg: dict = {}
    try:
        import jev_api
        cfg = {"accepted": jev_api.accepted(), "aliases": list(jev_api.JEV_ALIASES),
               "named": jev_api.named_ids(), "latest": jev_api.LATEST,
               "limits": {"choice_options": jev_api.MAX_CHOICE_OPTIONS,
                          "score_levels": [jev_api.MIN_SCORE_LEVELS,
                                           jev_api.MAX_SCORE_LEVELS],
                          "state_tokens": jev_api.STATE_LIMIT_TOKENS,
                          "request_tokens": jev_api.REQUEST_LIMIT_TOKENS},
               "one_call_at_a_time": True,
               "request_id_header": jev_api.REQUEST_ID_HEADER}
        mods = jev_api.models_list()
        cfg["models"] = (mods.get("x_yamadori") or {}).get("models")
    except Exception as e:                                       # noqa: BLE001
        cfg["error"] = f"{type(e).__name__}: {e}"[:200]
    return {"routes": routes, "unknown_routes": other,
            "requests": len(calls), "requested": requested, "served": served,
            "traffic": traffic, "errors": errors,
            "usage": {"input_tokens": inp, "output_tokens": outp,
                      "series": tok_series},
            "latency": spread([float(c["ms"]) for c in calls
                               if int(c.get("status") or 0) == 200
                               and c.get("questions")
                               and isinstance(c.get("ms"), (int, float))]),
            "recent": [{"ts": c["ts"], "route": c.get("route"),
                        "status": c.get("status"), "requested": c.get("requested"),
                        "model": c.get("model"), "traffic": c.get("traffic"),
                        "questions": len(c.get("questions") or {}),
                        "ms": c.get("ms"), "usage": c.get("usage"),
                        "error": (c.get("error") or {}).get("code")}
                       for c in calls[-25:]][::-1],
            "config": cfg,
            "note": ("every Jev call is a corpus event of kind jev_call "
                     "(mcp/jev_api.py). A 401 or a body that is not JSON, and "
                     "GET /jev/v1/models, are recorded since 2026-09-30.")}


def priors_section() -> dict:
    try:
        import decider_bonsai as D
        import jev_api
        models = jev_api.configured()
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    names = set(models) | {"bonsai", "mirai-s", "flash-next"}
    try:
        for f in os.listdir(D.PROFILES_DIR):
            if f.endswith(".json"):
                names.add(f[:-5])
    except OSError:
        pass
    out = {}
    for m in sorted(names):
        try:
            st = D.profile_status(m)
            meas = set(st.get("measured") or [])
            out[m] = {"configured": m in models, "record": bool(st.get("record")),
                      "measured": sorted(meas),
                      "unmeasured": st.get("unmeasured") or [],
                      "priors": {"letter_prior": "letter_prior" in meas,
                                 "label_bias": "label_bias" in meas},
                      "tie_band": st.get("tie_band"),
                      **({"error": st["error"]} if st.get("error") else {})}
        except Exception as e:                                   # noqa: BLE001
            out[m] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


def injector_section(rows: list[dict], reqs: list[dict]) -> dict:
    per: dict[str, dict] = {}
    skills_on = decider_turns = 0
    failures: dict[str, int] = {}
    for r in reqs:
        rec = r.get("rec") or {}
        if rec.get("skills_on"):
            skills_on += 1
        d = rec.get("decider")
        if isinstance(d, dict) and d.get("decisions"):
            decider_turns += 1
        if isinstance(d, dict) and isinstance(d.get("failure"), dict):
            k = str(d["failure"].get("code") or "?")
            failures[k] = failures.get(k, 0) + 1
        inj = rec.get("inject")
        if not isinstance(inj, dict):
            continue
        m = str(inj.get("model") or r.get("model") or "?")
        p = per.setdefault(m, {"model": m, "runs": 0, "injected": 0,
                               "skipped": 0, "stage1_items": [],
                               "stage1_skills": [], "stage2_passed": [],
                               "stage2_asked": [], "stage3": {},
                               "chosen": [], "failures": {}, "why": {},
                               "need": [], "noul": [], "ms": [],
                               "item_tiers": {}, "stage3_tiers": {}})
        p["runs"] += 1
        chosen = int(inj.get("chosen") or 0)
        p["injected" if chosen else "skipped"] += 1
        p["chosen"].append(chosen)
        for k_src, k_dst in (("stage1_items", "stage1_items"),
                             ("stage1_skills", "stage1_skills")):
            if isinstance(inj.get(k_src), (int, float)):
                p[k_dst].append(inj[k_src])
        items = inj.get("items") or []
        p["stage2_asked"].append(len(items))
        p["stage2_passed"].append(sum(1 for i in items if i.get("pass")))
        for i in items:
            if isinstance(i.get("need"), (int, float)):
                p["need"].append(float(i["need"]))
            t = str(i.get("tier") or "untuned")
            p["item_tiers"][t] = p["item_tiers"].get(t, 0) + 1
        s3 = inj.get("stage3")
        if isinstance(s3, dict):
            act = str(s3.get("act") or "?")
            p["stage3"][act] = p["stage3"].get(act, 0) + 1
            t = str(s3.get("tier") or "untuned")
            p["stage3_tiers"][t] = p["stage3_tiers"].get(t, 0) + 1
            if isinstance(s3.get("noul"), (int, float)):
                p["noul"].append(float(s3["noul"]))
        f = inj.get("failure")
        if isinstance(f, dict):
            k = str(f.get("code") or "?")
            p["failures"][k] = p["failures"].get(k, 0) + 1
        w = str(inj.get("why") or "")
        if w:
            p["why"][w] = p["why"].get(w, 0) + 1
        if isinstance(inj.get("ms"), (int, float)):
            p["ms"].append(float(inj["ms"]))
    models = []
    for p in per.values():
        mean = (lambda xs: round(sum(xs) / len(xs), 2) if xs else None)
        models.append({
            "model": p["model"], "runs": p["runs"], "injected": p["injected"],
            "skipped": p["skipped"],
            "stage1_items_mean": mean(p["stage1_items"]),
            "stage1_skills_mean": mean(p["stage1_skills"]),
            "stage2_asked_mean": mean(p["stage2_asked"]),
            "stage2_passed_mean": mean(p["stage2_passed"]),
            "stage3": p["stage3"], "chosen_mean": mean(p["chosen"]),
            "failures": p["failures"],
            "why": dict(sorted(p["why"].items(), key=lambda kv: -kv[1])[:8]),
            "need_hist": hist(p["need"]) if p["need"] else None,
            "noul_hist": hist(p["noul"]) if p["noul"] else None,
            "item_tiers": p["item_tiers"], "stage3_tiers": p["stage3_tiers"],
            "ms": spread(p["ms"])})
    qsets = [q for q in question_sets_section(
        [r for r in rows if family((r.get("question") or {}).get("name") or "")
         in ("skill_item", "skill_inject")])]
    summ = None
    try:
        import skill_inject
        s = skill_inject.summary()
        summ = {k: s.get(k) for k in ("version", "questions", "max_skills",
                                      "thresholds", "levels")}
        summ["profiles"] = {k: {kk: v.get(kk) for kk in ("family", "format",
                                                          "voice", "max_items")}
                            for k, v in (s.get("profiles") or {}).items()}
    except Exception as e:                                       # noqa: BLE001
        summ = {"error": f"{type(e).__name__}: {e}"[:200]}
    skills_tiers = None
    try:
        import tiers
        skills_tiers = {n: bool(t.get("skills")) for n, t in tiers.TIERS.items()}
    except Exception:                                            # noqa: BLE001
        pass
    return {"models": sorted(models, key=lambda m: -m["runs"]),
            "question_sets": qsets, "requests": len(reqs),
            "skills_on": skills_on, "decider_turns": decider_turns,
            "decider_failures": failures, "injector": summ,
            "skills_by_tier": skills_tiers,
            "note": ("the injector runs only where skills run; skills are OFF "
                     "at every tier (operator, 2026-09-29) unless a header "
                     "forces them. Stage 1 is the mechanical filter, stage 2 "
                     "a jjava Score per item (its `need`), stage 3 a jjava "
                     "noul over the shortlist (docs/JJAVA.md 5.1).")}


# -------------------------------------------------------------------- build --
def overview(window: str = DEFAULT_WINDOW, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    span, step = WINDOWS[window]
    n = int(span // step)
    since = (int(now // step) + 1) * step - n * step
    buckets = [since + i * step for i in range(n)]
    out: dict = {"window": {"name": window, "since": since, "until": now,
                            "bucket_s": step, "buckets": buckets,
                            "names": list(WINDOWS)},
                 "at": now}
    parts: dict = {}
    rows, dmeta = read_decisions(since)
    calls, cmeta = read_jev_calls(since)
    reqs = read_requests(since)
    out["sources"] = {"decisions": dmeta, "jev_calls": cmeta,
                      "stats": {k: v for k, v in stats_store.status().items()
                                if k != "db"},
                      "requests": len(reqs)}
    for name, fn in (
            ("usage", lambda: usage_section(rows, calls, since, step, n)),
            ("latency", lambda: latency_section(rows, calls, reqs, since, step, n)),
            ("question_sets", lambda: question_sets_section(rows)),
            ("thresholds", lambda: thresholds_section(rows)),
            ("lane", lambda: lane_section(since, step, n)),
            ("jev", lambda: jev_section(calls, since, step, n)),
            ("priors", priors_section),
            ("injector", lambda: injector_section(rows, reqs))):
        try:
            parts[name] = fn()
        except Exception as e:                                   # noqa: BLE001
            parts[name] = {"error": f"{name}: {type(e).__name__}: {e}"[:300]}
    out.update(parts)
    return out


def cached(window: str) -> dict:
    now = time.time()
    with _cache_lock:
        hit = _cache.get(window)
        if hit and now - hit[0] < CACHE_S:
            return hit[1]
    d = overview(window, now)
    with _cache_lock:
        _cache[window] = (now, d)
    return d


def handle_get(path: str):
    """(status, content_type, body) or None if the path is not ours."""
    p = path.rstrip("/")
    if p == "/dash/api/jjava":
        w = DEFAULT_WINDOW
    elif p.startswith("/dash/api/jjava/"):
        w = p[len("/dash/api/jjava/"):]
        if w not in WINDOWS:
            return _json(404, {"error": f"no window {w!r}: one of "
                                        f"{', '.join(WINDOWS)}"})
    else:
        return None
    try:
        return _json(200, cached(w))
    except Exception as e:                                       # noqa: BLE001
        return _json(500, {"error": f"jjava overview raised "
                                    f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    d = overview(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WINDOW)
    print(json.dumps({k: d[k] for k in ("window", "sources")}, indent=1,
                     default=str))
    print(json.dumps(d.get("usage", {}).get("totals"), indent=1))
