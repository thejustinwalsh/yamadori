#!/usr/bin/env python
"""Swap the STAGED dashboard bundle (web/dist-next) into place (web/dist).

WHY A STAGE. The live proxy serves web/dist from disk (mcp/dash_static.py):
a bundle built in place goes live at once, and a new SPA calls API routes
the running proxy may not have yet -- pages break while the operator uses
them (coordinator, 2026-09-30). So a build goes to web/dist-next
(`npm run build:stage` in web/), is verified there
(`python mcp/test_dash_static.py --staged`), and the deploy runs this ONE
step right after it restarts the proxy with the matching code:

    python scripts/swap_dash_dist.py            # dist-next -> dist
    python scripts/swap_dash_dist.py --check    # verify the stage, change nothing
    python scripts/swap_dash_dist.py --rollback # dist-prev -> dist

What it does: refuses unless web/dist-next is CURRENT (its .buildinfo
matches web/src: dash_static.check_buildinfo); moves web/dist to
web/dist-prev (the previous set-aside is removed first) and web/dist-next to
web/dist. Two directory renames; on Windows a rename can fail while a file
inside is open (a request being served), so each is retried for a few
seconds. If the second rename fails the first is undone. The running proxy
needs no restart for the files (it opens them per request); its startup
note on the bundle is refreshed at the next restart.

Exit 0 swapped (or --check passed), 1 refused, 2 the renames failed.
Afterwards commit web/dist (`git add -A web/dist`: the hashed asset names
change).
"""
from __future__ import annotations

import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

import dash_static  # noqa: E402

WEB = os.path.join(ROOT, "web")
DIST = os.path.join(WEB, "dist")
NEXT = os.path.join(WEB, "dist-next")
PREV = os.path.join(WEB, "dist-prev")
TRIES, WAIT_S = 20, 0.25       # a few seconds: a served file closes fast


def _rename(a: str, b: str) -> None:
    for i in range(TRIES):
        try:
            os.rename(a, b)
            return
        except OSError:
            if i == TRIES - 1:
                raise
            time.sleep(WAIT_S)


def check(stage: str = NEXT) -> dict:
    if not os.path.isdir(stage):
        return {"ok": False, "problems": [f"{stage} does not exist: run "
                                          "`npm run build:stage` in web/"]}
    return dash_static.check_buildinfo(stage)


def swap(web: str = WEB) -> int:
    dist, nxt, prev = (os.path.join(web, n) for n in ("dist", "dist-next", "dist-prev"))
    c = check(nxt)
    if not c["ok"]:
        print("  REFUSED: the staged bundle is not current")
        for p in c["problems"][:10]:
            print(f"    {p}")
        return 1
    if os.path.isdir(prev):
        shutil.rmtree(prev)
    moved = False
    try:
        if os.path.isdir(dist):
            _rename(dist, prev)
            moved = True
        _rename(nxt, dist)
    except OSError as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        if moved and not os.path.isdir(dist):
            _rename(prev, dist)
            print("  the previous bundle was put back")
        return 2
    print(f"  swapped: {nxt} -> {dist} ({c['checked']} source files current); "
          f"previous bundle at {prev}")
    return 0


def rollback(web: str = WEB) -> int:
    dist, prev = os.path.join(web, "dist"), os.path.join(web, "dist-prev")
    if not os.path.isdir(prev):
        print(f"  nothing to roll back to: {prev} does not exist")
        return 1
    tmp = os.path.join(web, "dist-rolled-back")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    try:
        if os.path.isdir(dist):
            _rename(dist, tmp)
        _rename(prev, dist)
    except OSError as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return 2
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  rolled back: {prev} -> {dist}")
    return 0


def main(argv: list[str]) -> int:
    if "--check" in argv:
        c = check()
        print(f"  {NEXT}: {'CURRENT' if c['ok'] else 'NOT current'} "
              f"({c.get('checked', 0)} source files)")
        for p in c["problems"][:10]:
            print(f"    {p}")
        return 0 if c["ok"] else 1
    if "--rollback" in argv:
        return rollback()
    return swap()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
