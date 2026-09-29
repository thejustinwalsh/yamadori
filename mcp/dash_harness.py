#!/usr/bin/env python
"""HARNESS TOOLS: the API behind the dashboard page (mcp/harness_kit.py).

Every path here is under /dash/api, which server.py gates with
accounts.identify before dispatching; an unauthenticated request never
reaches this module. The caller's account id (a hash prefix, never the key)
is recorded as the author of every write.

    GET  /dash/api/harness-kit                  the live entries, counts, the
                                                vocabularies (harnesses, kinds,
                                                statuses, proofs, wheres), the
                                                public base the export uses
    GET  /dash/api/harness-kit/deleted          the soft-deleted entries
    GET  /dash/api/harness-kit/entry/<id>       one entry with its history
    GET  /dash/api/harness-kit/export/<harness>       the kit, application/zip
    GET  /dash/api/harness-kit/export/<harness>.json  the same kit as JSON
                                                {manifest, files: {path: text}}
    POST /dash/api/harness-kit/entry            {fields} -> a new entry
    POST /dash/api/harness-kit/entry/edit       {id, base_version, fields}:
                                                a new version (409 if stale)
    POST /dash/api/harness-kit/entry/delete     {id, why}: soft delete
    POST /dash/api/harness-kit/entry/restore    {id}

The store is seeded from mcp/harness_kit_seed.py on the first request of a
process (idempotent: a seed item the store ever held is never re-added).
Errors are {ok: false, error, reasons} with 400 / 404 / 409.
"""
from __future__ import annotations

import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness_kit  # noqa: E402
import harness_kit_seed  # noqa: E402

PREFIX = "/dash/api/harness-kit"
POSTS = (PREFIX + "/entry", PREFIX + "/entry/edit", PREFIX + "/entry/delete", PREFIX + "/entry/restore")
_SEEDED: dict = {}
_SEED_LOCK = threading.Lock()


def _json(code: int, payload: dict):
    return code, "application/json", json.dumps(payload, default=str).encode()


def _refused(e: harness_kit.Refused):
    return _json(e.code, {"ok": False, "error": e.message, "reasons": e.reasons})


def ensure_seeded(db: str | None = None) -> dict:
    """Seed once per process and database path."""
    key = db or harness_kit.DB_PATH
    with _SEED_LOCK:
        if key not in _SEEDED:
            _SEEDED[key] = harness_kit.seed(harness_kit_seed.items(), db)
    return _SEEDED[key]


def window_source() -> tuple[int | None, int | None]:
    """The proxy's advertised window and output ceiling (/v1/models,
    mcp/catalog.py), or (None, None) when the model server cannot say."""
    try:
        import catalog
        w = catalog.context_window()
        return int(w), int(catalog.max_output(w))
    except Exception:                                            # noqa: BLE001
        return None, None


def vocab() -> dict:
    return {"harnesses": [{"id": h, "label": lbl} for h, lbl in harness_kit.HARNESSES],
            "kinds": [{"id": k, "label": lbl} for k, lbl in harness_kit.KINDS],
            "statuses": list(harness_kit.STATUSES),
            "proofs": [{"id": p, "label": lbl} for p, lbl in harness_kit.PROOFS],
            "wheres": [{"id": w, "label": lbl} for w, lbl in harness_kit.WHERES],
            "skill_roots": list(harness_kit.SKILL_ROOTS),
            "tokens": list(harness_kit.TOKENS),
            "key_placeholder": harness_kit.KEY_PLACEHOLDER}


def build_kit(harness: str, db: str | None = None) -> dict:
    rows = harness_kit.entries(db)
    w, mo = window_source()
    return harness_kit.export(harness, rows, base=harness_kit.public_base(), window=w, max_output=mo)


def handle_get(path: str, db: str | None = None):
    path = path.rstrip("/")
    if path != PREFIX and not path.startswith(PREFIX + "/"):
        return None
    try:
        seeded = ensure_seeded(db)
        if path == PREFIX:
            rows = harness_kit.entries(db)
            return _json(200, {"ok": True, "entries": rows, "counts": harness_kit.counts(rows),
                               "public_base": harness_kit.public_base(),
                               "seed": {"version": harness_kit_seed.SEED_VERSION,
                                        "this_process": {k: len(v) for k, v in seeded.items()}},
                               **vocab()})
        if path == PREFIX + "/deleted":
            rows = [r for r in harness_kit.entries(db, include_deleted=True) if r["deleted"]]
            return _json(200, {"ok": True, "entries": rows})
        if path.startswith(PREFIX + "/entry/"):
            e = harness_kit.get(path[len(PREFIX + "/entry/"):], db)
            if e is None:
                return _json(404, {"ok": False, "error": "no such entry"})
            return _json(200, {"ok": True, "entry": e})
        if path.startswith(PREFIX + "/export/"):
            name = path[len(PREFIX + "/export/"):]
            as_json = name.endswith(".json")
            harness = name[:-5] if as_json else name
            if harness not in harness_kit.HARNESS_IDS:
                return _json(404, {"ok": False, "error": f"no harness {harness!r}"})
            kit = build_kit(harness, db)
            if as_json:
                return _json(200, {"ok": True, "manifest": kit["manifest"], "top": kit["top"],
                                   "files": {p: b.decode("utf-8", "replace") for p, b in kit["files"].items()}})
            return 200, "application/zip", harness_kit.zip_bytes(kit)
        return _json(404, {"ok": False, "error": "not found"})
    except harness_kit.Refused as e:
        return _refused(e)


def handle_post(path: str, body: dict, who: str = "operator", db: str | None = None):
    path = path.rstrip("/")
    if path not in POSTS:
        return None
    author = f"operator:{who}" if who else "operator"
    if not isinstance(body, dict):
        return _json(400, {"ok": False, "error": "the body is a JSON object"})
    try:
        ensure_seeded(db)
        if path == PREFIX + "/entry":
            fields = body.get("fields") if isinstance(body.get("fields"), dict) else body
            return _json(200, {"ok": True, "entry": harness_kit.create(fields, author, db)})
        eid = str(body.get("id") or "")
        if not eid:
            return _json(400, {"ok": False, "error": "id is required", "reasons": [{"field": "id", "why": "required"}]})
        if path == PREFIX + "/entry/edit":
            fields = body.get("fields")
            if not isinstance(fields, dict):
                return _json(400, {"ok": False, "error": "fields is required",
                                   "reasons": [{"field": "fields", "why": "an object"}]})
            return _json(200, {"ok": True, "entry": harness_kit.edit(eid, fields, author, body.get("base_version"), db)})
        if path == PREFIX + "/entry/delete":
            return _json(200, {"ok": True, "entry": harness_kit.set_deleted(eid, True, author, str(body.get("why") or ""), db)})
        return _json(200, {"ok": True, "entry": harness_kit.set_deleted(eid, False, author, "", db)})
    except harness_kit.Refused as e:
        return _refused(e)
