#!/usr/bin/env python
"""The MCP servers the proxy hosts for the model, for the dashboard.

Under /dash/api, which server.py gates with accounts.identify before
dispatching; an unauthenticated request never reaches this module.

    GET /dash/api/mcp    the configuration in force (mcp/mcp_config.py: the
                         file, or the built-in default), every server's row
                         -- runtime, package, image, pin, network, its
                         call_timeout_s and why, the tools of ours it backs
                         with their descriptions and declared overlaps, the
                         system line -- and its state in this process
                         (mcp/mcp_host.py: stopped / starting / ready /
                         exited / failed, why, starts, the server's own
                         name and version, the upstream tools it listed)

Read-only for now (operator, 2026-09-28: a page "like skills" to add,
enable and disable servers comes later; mcp_config.save / set_enabled are
its writers, built and tested, not routed). No secret is held by any of it.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcp_host  # noqa: E402


def _json(code: int, payload: dict):
    return code, "application/json", json.dumps(payload, default=str).encode()


def handle_get(path: str):
    if path.rstrip("/") != "/dash/api/mcp":
        return None
    try:
        return _json(200, mcp_host.status())
    except Exception as e:                                       # noqa: BLE001
        return _json(500, {"error": f"mcp status raised "
                                    f"{type(e).__name__}: {e}"})
