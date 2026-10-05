#!/usr/bin/env python
"""WHAT RUNS WHERE, for the dashboard: every card, every model on it, every slot and job lane, and what holds what.

The operator, 2026-10-05, verbatim: "the dashboard is not showing me the other agent lanes at all, some weird line
about someone else holding the GPU, or something. I want to know the status of anything running in the bargraph, we
should be swapping out the loadouts if that is what we need, Flash-Next is not showing anything when running in the
dashboard."

ONE READ-ONLY SNAPSHOT (`snapshot()`, carried as `cards` on /dash/api/vitals and /dash/api/vitals/pulse):

  cards[]   one per GPU (the 5060 Ti `main`, the A4000 `a4000`): the models llama-swap has loaded on it
            (`/running`: model, state, port, gguf, ttl), each with ITS OWN slots read from its own port (never
            /upstream/<model>/, which would load an unloaded one: docs/DASHBOARD.md "a view never loads a model"),
            what each slot is for (`use`), the requests the proxy holds in flight on it and since when, and the
            models the tier table expects there that are not loaded (`expected`: bonsai-a4000 beside a locked
            main model)
  jobs[]    the worker's GPU job lanes (mcp/jobs.py GPU_SCOPES: `gpu` the main card, `gpu_a4000` the A4000): paused
            (by, why, since, until), the job running now, the jobs waiting
  holds[]   what holds what, each in plain words with its reason and since when: a paused job lane, a model lease
            (requests in flight), a swap in progress, a conversation's hold of a card (slots.holds), an A4000 lease
            (gpu_room.leases), a model llama-swap is starting or stopping
  swap      the swap in progress (max_mode.snapshot `swap_now`: from, to, phase, since) and the last swaps (the
            persisted ones, stats_store `swaps`, newest first, and the in-memory last one)

Only the proxy process knows the leases, the swap in flight and the conversation holds (max_mode, slots are
in-memory); anywhere else those parts are empty and `in_proxy` says so. Nothing here writes, loads or asks a model
that /running does not list as ready. Every read is bounded (a /slots read waits at most vitals.SLOTS_TIMEOUT) and
a failed part says so by itself without failing the rest.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
import json

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import vitals  # noqa: E402

SWAPS_SHOWN = 5
SWAP_HISTORY_S = 7 * 86400
# a /slots read of a model that is not generating waits for nothing; the slowest allowed here
SLOTS_TIMEOUT = vitals.SLOTS_TIMEOUT

_lock = threading.Lock()
# when this process FIRST SAW a model in a transitional state (starting / stopping): llama-swap does not say
_first_seen: dict[tuple[str, str], float] = {}


def _a4000_uuid() -> str:
    try:
        import gpu_room
        return str(gpu_room.CARD_UUID).lower()
    except Exception:                                            # noqa: BLE001
        return ""


def card_of(model: str, room=None) -> str:
    """Which card a loaded model sits on: `main` (a main model of the tier table, or the default), `a4000` (a model
    gpu_room sizes for the A4000: the embedder, bonsai-a4000, vision, the image models), else `other`."""
    if model in vitals.main_models():
        return "main"
    try:
        if room is None:
            import gpu_room as room
        if room.on_card(model) or model == "bonsai-a4000":
            return "a4000"
    except Exception:                                            # noqa: BLE001
        pass
    return "other"


def _is_llama_server(cmd: str) -> bool:
    c = (cmd or "").lower()
    return "llama-server" in c and "--embeddings" not in c


def _use_of(model: str, slot: dict, card: str, holds: dict | None) -> str | None:
    """What a slot is for, in the layout the code runs (docs: AGENTS.md "Layout v3")."""
    sid = slot.get("id")
    if card == "main":
        if slot.get("role") == "child":
            return "jjava lane (the decider, side calls)"
        return "conversation (primary)" if slot.get("primary") else "conversation"
    try:
        import slots as slot_map
        conv, lane = slot_map.OTHER_CONV_SLOT, slot_map.OTHER_LANE_SLOT
    except Exception:                                            # noqa: BLE001
        conv, lane = 0, 1
    if model == "bonsai-a4000":
        if sid == conv:
            who = (holds or {}).get("other") or {}
            return "second conversation (the other card)" + (f", {who['owner']}" if who.get("owner") else "")
        if sid == lane:
            return "jjava, side calls"
    if model == "bonsai-vision":
        return "image reads (vision)"
    return None


def _conv_of(model: str, slot_id, card: str, occ: dict | None) -> dict | None:
    """The conversation pinned to this slot (slots.occupants): {conv, busy, idle_s, held, rest_s, hold_s}."""
    if not occ:
        return None
    if card == "main":
        c = (occ.get("main") or {}).get(slot_id)
    else:
        o = occ.get("other") or {}
        c = o if (model == occ.get("other_model") or model == "bonsai-a4000") and o.get("slot") == slot_id else None
    return dict(c, hold_s=occ.get("hold_s")) if c else None


def _read_slots(model: str, url: str, card: str, holds: dict | None, occ: dict | None = None) -> dict:
    """One model's /slots from its OWN port. {ok, slots[], ms, error?}; rates per (model, slot)."""
    t0 = time.time()
    try:
        with urllib.request.urlopen(vitals.local_url(url) + "/slots", timeout=SLOTS_TIMEOUT) as r:
            data = json.load(r)
    except Exception as e:                                       # noqa: BLE001
        return {"ok": False, "slots": [], "error": f"{type(e).__name__}: {e}"[:200],
                "ms": round((time.time() - t0) * 1000)}
    if not isinstance(data, list):
        return {"ok": False, "slots": [], "error": "unexpected /slots shape",
                "ms": round((time.time() - t0) * 1000)}
    now = time.time()
    with vitals._lock:
        rows = [vitals._slot_row(s, now, ns=model) for s in data if isinstance(s, dict)]
    if card == "main":
        vitals._slot_roles(rows)
    for r in rows:
        r["use"] = _use_of(model, r, card, holds)
        r["conv"] = _conv_of(model, r["id"], card, occ)
    return {"ok": True, "slots": rows, "ms": round((now - t0) * 1000),
            "decoding": sum(1 for r in rows if r["state"] == "decode"),
            "prefilling": sum(1 for r in rows if r["state"] == "prefill"),
            "tps": round(sum(r["tps"] for r in rows), 1)}


def _gpus(given: list | None) -> list[dict]:
    """The cards, as the dashboard already reads them (the proxy's power sampler's own 1 s read, else nvidia-smi);
    the caller's own read when it has one."""
    if given:
        return given
    g = vitals._sampled_gpus(2 * vitals.PULSE_GPU_TTL)
    return g if g is not None else vitals.gpus(timeout=3)


def _queue_by_scope() -> dict:
    """The worker's gpu queue by scope, read only: {scope: {running: [job], queued: n}}."""
    out = {"gpu": {"running": [], "queued": 0}, "gpu_a4000": {"running": [], "queued": 0}}
    try:
        import jobs
        con = vitals._ro(jobs.DB)
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    try:
        off = jobs.internal_off_main()
        now = time.time()
        for jid, queue, started, ds, st, lane, card, hb in con.execute(
                "SELECT id, queue, started, dataset, stage, lane, card, heartbeat FROM jobs "
                "WHERE state='running' AND lane IN ('gpu','gpu_a4000')"):
            scope = card or lane
            out.setdefault(scope, {"running": [], "queued": 0})["running"].append(
                {"id": jid, "queue": queue, "dataset": ds, "stage": st, "started": started, "heartbeat": hb})
        for queue, payload, lane in con.execute(
                "SELECT queue, payload, lane FROM jobs WHERE state='queued' AND lane IN ('gpu','gpu_a4000') "
                "AND (not_before IS NULL OR not_before <= ?)", (now,)):
            try:
                scope = jobs.gpu_scope(queue, jobs._loads(payload), lane, off)
            except Exception:                                    # noqa: BLE001
                scope = "gpu"
            out.setdefault(scope, {"running": [], "queued": 0})["queued"] += 1
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    finally:
        con.close()
    return out


def _last_swaps(mm, now: float) -> list[dict]:
    rows: list[dict] = []
    try:
        import stats_store
        for r in stats_store.read("swaps", now - SWAP_HISTORY_S)[-SWAPS_SHOWN:][::-1]:
            try:
                frm = json.loads(r.get("from_models") or "[]")
                left = json.loads(r.get("left_loaded") or "[]")
            except ValueError:
                frm, left = [], []
            rows.append({"at": r["ts"], "from": frm, "to": r.get("to_model"), "load_s": r.get("load_s"),
                         "ok": bool(r.get("ok")), "how": r.get("how"), "left_loaded": left})
    except Exception:                                            # noqa: BLE001
        pass
    # the in-memory last swap is newer than the persisted rows when the writer thread is behind: add it once
    try:
        last = (mm.snapshot() or {}).get("last_swap") if mm is not None else None
    except Exception:                                            # noqa: BLE001
        last = None
    if last and last.get("at") and not any(abs(float(r["at"]) - float(last["at"])) < 5 for r in rows):
        rows.insert(0, {"at": last.get("at"), "from": last.get("from") or [], "to": last.get("to"),
                        "load_s": last.get("load_s"), "ok": bool(last.get("ok")), "how": last.get("how"),
                        "left_loaded": last.get("left_loaded") or []})
    return rows[:SWAPS_SHOWN]


def _hold(kind: str, card: str, what: str, *, by=None, why=None, since=None, until=None, **more) -> dict:
    return {"kind": kind, "card": card, "what": what, "by": by, "why": why, "since": since, "until": until, **more}


def snapshot(gpus: list | None = None) -> dict:
    """The whole view. Cheap: /running (cached 2 s), one /slots per loaded llama-server (<= SLOTS_TIMEOUT), one
    read-only jobs query, files."""
    now = time.time()
    mm = vitals._max_mode()
    in_proxy = "server" in sys.modules or "proxy" in sys.modules
    rows = vitals.running_rows()
    try:
        import slots as slot_map
        sh = slot_map.holds()
        occ = slot_map.occupants()
    except Exception:                                            # noqa: BLE001
        sh, occ = None, None
    try:
        msnap = mm.snapshot() if mm is not None else {}
    except Exception:                                            # noqa: BLE001
        msnap = {}
    inflight = dict(msnap.get("inflight") or {})
    since_of = dict(msnap.get("inflight_since") or {})
    try:
        import gpu_room
        room_use = gpu_room._load().get("last_use", {}) if gpu_room.enabled() else {}
        leases = gpu_room.leases() if gpu_room.enabled() else []
    except Exception:                                            # noqa: BLE001
        room_use, leases = {}, []

    gpus = _gpus(gpus)
    a_uuid = _a4000_uuid()
    cards: list[dict] = []
    for g in gpus:
        key = ("a4000" if str(g.get("uuid") or "").lower() == a_uuid
               else "main" if g.get("main") else f"gpu{g.get('index')}")
        cards.append({"key": key, "index": g.get("index"), "name": g.get("name"), "uuid": g.get("uuid"),
                      "util": g.get("util"), "used_mib": g.get("used_mib"), "total_mib": g.get("total_mib"),
                      "free_mib": g.get("free_mib"), "models": [], "expected": []})
    by_key = {c["key"]: c for c in cards}

    loaded_names: set[str] = set()
    todo: list[tuple[dict, dict, str]] = []
    for r in rows or []:
        m = str(r.get("model"))
        state = str(r.get("state") or "ready")
        if state == "stopped":
            continue
        loaded_names.add(m)
        ck = card_of(m)
        entry = {"model": m, "state": state, "port": vitals._port_of(str(r.get("proxy") or ""), 0) or None,
                 "gguf": vitals._gguf(str(r.get("cmd") or "")), "ttl": r.get("ttl"), "card": ck,
                 "main": ck == "main", "slots": None, "inflight": int(inflight.get(m) or 0),
                 "inflight_since": since_of.get(m), "last_use": room_use.get(m) if ck == "a4000" else None}
        card = by_key.get(ck)
        if card is None:
            card = by_key.setdefault(ck, {"key": ck, "index": None, "name": "unplaced", "uuid": None, "models": [],
                                         "expected": []})
            if card not in cards:
                cards.append(card)
        card["models"].append(entry)
        with _lock:
            if state in ("starting", "stopping"):
                _first_seen.setdefault((m, state), now)
            else:
                for k in [k for k in _first_seen if k[0] == m]:
                    _first_seen.pop(k, None)
        if state == "ready" and r.get("proxy") and _is_llama_server(str(r.get("cmd") or "")):
            todo.append((entry, r, ck))

    # every model's slots, each from its own port, in parallel (a slow one costs one timeout, not a sum)
    def read(entry: dict, r: dict, ck: str) -> None:
        entry["slots"] = _read_slots(entry["model"], str(r["proxy"]), ck, sh, occ)
    threads = [threading.Thread(target=read, args=t, daemon=True) for t in todo]
    for t in threads:
        t.start()
    for t in threads:
        t.join(SLOTS_TIMEOUT + 1.0)
    for entry, _r, _ck in todo:
        if entry["slots"] is None:
            entry["slots"] = {"ok": False, "slots": [], "error": "the read did not finish in time"}

    # what the tier table expects on the A4000 beside the main model that is loaded and is LOCKED
    expected: list[dict] = []
    if mm is not None and getattr(mm, "ENABLED", False):
        cr = vitals.card_row(rows) if rows is not None else None
        on = str(cr.get("model")) if cr else None
        if on and mm.TABLE.locked(on):
            h = mm.TABLE.helpers(on) or {}
            for role in ("decider", "side_calls"):
                hm = h.get(role)
                if hm and hm != "self" and hm not in loaded_names and not any(e["model"] == hm for e in expected):
                    expected.append({"model": hm, "for": f"{on}'s jjava and side calls (the card runs {on} alone)",
                                     "why": f"{hm} is not loaded; llama-swap loads it on the first such request"})
    if "a4000" in by_key:
        by_key["a4000"]["expected"] = expected

    # ---- the swap
    swap_now = msnap.get("swap_now")
    last_swaps = _last_swaps(mm, now)

    # ---- the holds, in plain words
    holds: list[dict] = []
    if rows is None:
        holds.append(_hold("unknown", "main", "llama-swap /running could not be read, so what is loaded is unknown"))
    if swap_now:
        frm = ", ".join(swap_now.get("from") or []) or "nothing"
        holds.append(_hold(
            "swap", "main",
            (f"waiting for the work in flight on {frm} to end before loading {swap_now.get('to')}"
             if swap_now.get("phase") == "waiting" else
             f"llama-swap is loading {swap_now.get('to')} onto the 5060 Ti (replacing {frm})"),
            by="the proxy", since=swap_now.get("since"), to=swap_now.get("to")))
    for c in cards:
        for e in c["models"]:
            if e["state"] in ("starting", "stopping"):
                with _lock:
                    since = _first_seen.get((e["model"], e["state"]))
                holds.append(_hold("model_state", c["key"], f"llama-swap: {e['model']} is {e['state']}",
                                   by="llama-swap", since=since, since_basis="first seen by this view"))
            if e["inflight"] > 0:
                holds.append(_hold(
                    "model_lease", c["key"],
                    f"{e['inflight']} request{'s' if e['inflight'] != 1 else ''} in flight on {e['model']}",
                    by="the proxy (max_mode lease)", since=e["inflight_since"]))
    for m, n in inflight.items():          # a lease for a model that is not loaded: the request is waiting for it
        if int(n or 0) > 0 and m not in loaded_names:
            holds.append(_hold("model_lease", card_of(m),
                               f"{n} request{'s' if int(n) != 1 else ''} waiting for {m}, which is not loaded yet",
                               by="the proxy (max_mode lease)", since=since_of.get(m)))
    if sh:
        for ck, label in (("main", "main card"), ("other", "other card")):
            v = sh.get(ck)
            if v:
                tgt = "main" if ck == "main" else "a4000"
                if v["busy"]:
                    what = f"conversation {v['owner']} has a request in flight on the {label}"
                else:
                    what = (f"conversation {v['owner']} holds the {label} for {v['rest_s']} more s "
                            f"(idle {v['idle_s']:.0f} s of its {v['hold_s']:.0f} s hold); another conversation "
                            f"{'goes to the other card' if ck == 'main' else 'waits'} until then")
                holds.append(_hold("conversation", tgt, what, by="the proxy (slots)",
                                   since=(now - v["idle_s"]) if not v["busy"] else None,
                                   until=(now + v["rest_s"]) if not v["busy"] else None))
    for ls in leases:
        holds.append(_hold("gpu_room_lease", "a4000", f"a request holds {ls['model']} on the A4000",
                           by=f"pid {ls.get('pid')} (gpu_room lease)", since=ls.get("since"), until=ls.get("until")))
    pauses = vitals._jobs_paused()
    q = _queue_by_scope()
    jobs_view: list[dict] = []
    for lane, label, ck in (("gpu", "main card (5060 Ti)", "main"), ("gpu_a4000", "A4000", "a4000")):
        p = next((x for x in pauses if x["lane"] == lane), None)
        qs = q.get(lane) if isinstance(q, dict) else None
        jobs_view.append({"lane": lane, "card": ck, "label": label, "paused": p,
                          "running": (qs or {}).get("running", []), "queued": (qs or {}).get("queued"),
                          "error": q.get("error") if isinstance(q, dict) else None})
        if p:
            run = len((qs or {}).get("running") or [])
            holds.append(_hold(
                "job_lane_paused", ck,
                f"the worker's {lane} job lane ({label}) is paused: no new job starts there"
                + (f"; {run} already running finishes" if run else "")
                + ". This is a pause on the worker's queue, not a hold on the card",
                by=p.get("by"), why=p.get("why"), since=p.get("since"), until=p.get("until")))

    return {"at": now, "in_proxy": in_proxy, "running_read": rows is not None, "cards": cards, "jobs": jobs_view,
            "holds": holds, "swap": {"now": swap_now, "last": last_swaps}}
