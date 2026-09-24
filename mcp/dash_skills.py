#!/usr/bin/env python
"""The skills API behind the dashboard: list, view, add, edit, disable,
re-enable, watch.

Every path here is under /dash/api, which `server.py` gates with
`accounts.identify` before dispatching -- an unauthenticated request never
reaches this module. The caller's account id (a hash prefix, never the key)
is recorded as the author of an edit or a disable.

    GET  /dash/api/skills            every skill, the recall path, the limits
    GET  /dash/api/skills/<id>       one skill with its versions and jobs
    POST /dash/api/skill             {url} or {text, name?}, optional
                                     watch_hours: ingest (arms on its own)
    POST /dash/api/skill/edit        {id, text}: a new version, re-screened
                                     before it re-arms; the skill serves
                                     nothing meanwhile
    POST /dash/api/skill/disable     {id, reason?}
    POST /dash/api/skill/enable      {id}: the served version serves again
    POST /dash/api/skill/watch       {id, hours}: 0 or null stops watching
    POST /dash/api/skill/refetch     {id}: enqueue a watch now

Nothing here runs a stage: every write records a row and enqueues a job for
`mcp/worker.py`, exactly as datasets.py does. No response carries a
filesystem path (skills.public).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402
import skill_learn  # noqa: E402
import skill_limits  # noqa: E402
import skill_select  # noqa: E402
import skills  # noqa: E402

POSTS = ("/dash/api/skill", "/dash/api/skill/edit", "/dash/api/skill/disable",
         "/dash/api/skill/enable", "/dash/api/skill/watch",
         "/dash/api/skill/refetch")


def _json(code: int, payload: dict):
    return code, "application/json", json.dumps(payload, default=str).encode()


def overview() -> dict:
    return {
        "skills": [skills.public(s) for s in skills.listing()],
        "counts": skills.counts(),
        "recall": skill_select.recall_path(),
        "skip_classes": list(skill_select.SKIP_CLASSES),
        "stages": list(skills.STAGES),
        "limits": skill_limits.summary(),
        "thresholds": {"emb_high": skill_select.EMB_HIGH,
                       "emb_low": skill_select.EMB_LOW,
                       "label": "unmeasured starting points"},
        # The fallback rate per day, pending records, and whether the stack
        # is idle enough to learn from them (skill_learn).
        "learning": skill_learn.overview(),
        "queue": {k: v for k, v in jobs.snapshot().items() if k != "db"},
    }


def handle_get(path: str):
    """(status, content_type, body) or None if this path is not ours."""
    p = path.rstrip("/")
    if p == "/dash/api/skills":
        try:
            return _json(200, overview())
        except Exception as e:                                   # noqa: BLE001
            return _json(500, {"error": f"skills overview raised "
                                        f"{type(e).__name__}: {e}"})
    if p.startswith("/dash/api/skills/"):
        sid = p.rsplit("/", 1)[-1]
        s = skills.get(sid)
        if s is None:
            return _json(404, {"error": f"no such skill: {sid}"})
        return _json(200, {"skill": skills.public(s, detail=True)})
    return None


def handle_post(path: str, body: dict, who: str = "operator"):
    p = path.rstrip("/")
    if p not in POSTS:
        return None
    body = body or {}
    author = f"operator:{who}" if who else "operator"
    try:
        if p == "/dash/api/skill":
            url = str(body.get("url") or "").strip() or None
            text = body.get("text")
            text = str(text) if isinstance(text, str) and text.strip() else None
            s = skills.create(url=url, text=text, name=body.get("name"),
                              author=author,
                              watch_hours=body.get("watch_hours"))
            return _json(200, {"ok": True, "skill": skills.public(s)})
        sid = str(body.get("id") or "")
        if p == "/dash/api/skill/edit":
            s = skills.edit(sid, str(body.get("text") or ""), author=author)
        elif p == "/dash/api/skill/disable":
            s = skills.disable(sid, reason=str(body.get("reason") or ""),
                               author=author)
        elif p == "/dash/api/skill/enable":
            s = skills.enable(sid, author=author)
        elif p == "/dash/api/skill/watch":
            h = body.get("hours")
            s = skills.set_watch(sid, float(h) if h not in (None, "") else None)
        else:
            s = skills.get(sid)
            if s is None:
                raise KeyError(f"no such skill: {sid}")
            if not s.get("source_url"):
                raise ValueError("only a skill with a source URL can be "
                                 "re-fetched")
            jid = jobs.add(skills.WATCH[0], {"skill": sid},
                           lane=skills.WATCH[1], dataset=f"skill:{sid}",
                           stage="watch")
            return _json(200, {"ok": True, "job": jid,
                               "skill": skills.public(s)})
        return _json(200, {"ok": True, "skill": skills.public(s)})
    except KeyError as e:
        return _json(404, {"ok": False, "error": str(e).strip("'\"")})
    except Exception as e:                                       # noqa: BLE001
        return _json(400, {"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    o = overview()
    print(f"  {len(o['skills'])} skill(s); {o['counts']}; recall path "
          f"{o['recall']}")
