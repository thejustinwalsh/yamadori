"""THE STACK'S DOCKER IS THE WSL ENGINE, whatever the user's current context says.

2026-10-15: after the Windows Update reboot the docker client's `currentContext` was `desktop-linux` again (Docker
Desktop resets it when it starts; docs/DOCKER-WSL.md set it to `wsl-ubuntu`), so `docker` reached a Docker Desktop
that is not running: PackageLens failed, and every new conversation at medium and up waited 62 s for
scripts/wsl_engine.ps1 (deploy check, x_yamadori overhead 62.5 s at medium/high/max). Every module of ours that runs
docker (mcp_host, typecheck) calls `pin()` at import, so this process and every docker subprocess it starts use
YAMADORI_DOCKER_CONTEXT, default `wsl-ubuntu`, when that context exists and neither DOCKER_CONTEXT nor DOCKER_HOST
is set. YAMADORI_WSL_ENGINE=0 (a machine whose Docker is not the WSL engine) leaves the user's context alone.
"""
from __future__ import annotations

import hashlib
import os

DEFAULT = "wsl-ubuntu"


def pin() -> str | None:
    """The context pinned for this process, or None (already set, not wanted, or not on this machine)."""
    if os.environ.get("DOCKER_CONTEXT") or os.environ.get("DOCKER_HOST") or \
            os.environ.get("YAMADORI_WSL_ENGINE") == "0":
        return None
    ctx = os.environ.get("YAMADORI_DOCKER_CONTEXT") or DEFAULT
    meta = os.path.join(os.environ.get("DOCKER_CONFIG") or os.path.join(os.path.expanduser("~"), ".docker"),
                        "contexts", "meta", hashlib.sha256(ctx.encode()).hexdigest())
    if not os.path.isdir(meta):
        return None
    os.environ["DOCKER_CONTEXT"] = ctx
    return ctx
