#!/usr/bin/env python
"""Deep thinking's triggers and learner, for the dashboard (Phase 0.6).

Under /dash/api, which server.py gates with accounts.identify before
dispatching; an unauthenticated request never reaches this module.

    GET  /dash/api/deep          thresholds in force (default, pinned by env,
                                 or learned, with the adjustment and its n),
                                 bounds, adjustments, description proposals,
                                 runs per day by trigger, label counts,
                                 recent decisions, and whether the learner
                                 is idle enough to run
    POST /dash/api/deep/revert   {id}: undo one adjustment (recorded as a new
                                 adjustment naming it; the caller's account
                                 is the author)
    POST /dash/api/deep/e1/revert  {head, version?}: serve an older E1 head
                                 version (default: the one before the
                                 current; 0 = none, the rule decides)
    POST /dash/api/deep/e1/decide  {head, question, context?}: the served E1
                                 head's answer for one state -- choice,
                                 probabilities, version, n (no text back).
                                 The live check (mcp/test_live_stack.py
                                 `e1`) compares it with the offline run.

GET /dash/api/deep carries `e1`: every head, its current version and every
version with n, CV accuracy, status and evaluation (e1.overview).

The panel that draws it can follow later (operator, 2026-09-24). No response
carries message text: rows hold names, counts and labels.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deep_learn  # noqa: E402


def _json(code: int, payload: dict):
    return code, "application/json", json.dumps(payload, default=str).encode()


def handle_get(path: str):
    if path.rstrip("/") != "/dash/api/deep":
        return None
    try:
        return _json(200, deep_learn.overview())
    except Exception as e:                                       # noqa: BLE001
        return _json(500, {"error": f"deep overview raised "
                                    f"{type(e).__name__}: {e}"})


def _e1_post(path: str, body: dict, who: str):
    import e1
    body = body or {}
    head = str(body.get("head") or "")
    if head not in e1.HEADS:
        return _json(400, {"ok": False, "error": f"`head` is one of "
                                                 f"{sorted(e1.HEADS)}"})
    if path.endswith("/revert"):
        v = body.get("version")
        out = e1.revert(head, int(v) if v is not None else None,
                        author=f"operator:{who}" if who else "operator")
        return _json(200 if out.get("ok") else 404, out)
    if e1.HEADS[head].get("pair"):
        return _json(400, {"ok": False, "error": "skill_applies needs a skill;"
                                                 " not probed here"})
    d, status = e1.decide(head, str(body.get("question") or ""),
                          str(body.get("context") or ""))
    return _json(200, {"ok": d is not None, "status": status,
                       "enabled": e1.enabled(), "decision": d})


def handle_post(path: str, body: dict, who: str = "operator"):
    p = path.rstrip("/")
    if p in ("/dash/api/deep/e1/revert", "/dash/api/deep/e1/decide"):
        try:
            return _e1_post(p, body, who)
        except Exception as e:                                   # noqa: BLE001
            return _json(500, {"ok": False, "error": f"{type(e).__name__}: "
                                                     f"{e}"})
    if p != "/dash/api/deep/revert":
        return None
    aid = str((body or {}).get("id") or "")
    if not aid:
        return _json(400, {"ok": False, "error": "`id` names the adjustment "
                                                 "to revert"})
    out = deep_learn.revert(aid, author=f"operator:{who}" if who
                            else "operator")
    return _json(200 if out.get("ok") else 404, out)
