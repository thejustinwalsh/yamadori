#!/usr/bin/env python
"""Serve the React dashboard (web/dist) at the site root, and prove it is current.

WHY THE BUNDLE IS COMMITTED

Node is a contributor dependency, never a runtime one (design/STACK.md). The
built bundle is committed and served as static files, so someone self-hosting
this runs Python exactly as before and never meets npm.

WHY IT CARRIES A HASH OF ITS SOURCE

Committed build output goes stale silently: a .tsx is edited, nobody
rebuilds, and the repo ships a dashboard that is not what its source says.
Nothing errors. `web/build/buildinfo.ts` writes `web/dist/.buildinfo` -- the
sha256 of every source file that fed the build -- and `check_buildinfo()`
recomputes them here, in Python, in milliseconds, with no Node. A stale bundle
is logged loudly at startup and FAILS `mcp/test_dash_static.py`.

The hash is over the bytes with CRLF normalised to LF, exactly as the Node
writer does it, or a Windows checkout with autocrlf would report every file
stale -- and a check that cries wolf gets ignored.

THREE MODES, ONE DECISION (`ui_mode()`)

  react   dist/index.html exists: the SPA owns the site root. `/` and any
          unknown GET that asks for HTML get index.html (client routes such
          as /nebari or /data/<id> survive a refresh), /assets/* are the
          hashed chunks, and old /dash and /dash/<route> bookmarks 302 to
          the root equivalent. /dash/api/* stays the JSON API and the Python
          pages stay at /dash/classic*, where they still own the two write
          flows the React build does not have yet: recipe review
          (/dash/classic) and dataset submission (/dash/classic/data).
  python  YAMADORI_DASH_UI=python, or no bundle: exactly the old behaviour.
  dev     --dev or YAMADORI_DASH_DEV=1: dist is NOT served, and an HTML
          request says where Vite is. The browser talks to Vite on :5173 and
          Vite proxies /dash/api to here -- never the reverse, because
          relaying HMR's websocket through Starlette degrades to silent full
          reloads.

WHAT THE SPA NEVER ANSWERS

The fallback is a catch-all, so it is registered LAST and refuses, as a
plain 404, anything under a server prefix (RESERVED): /v1, /health,
/dash/api, /dash/classic, /assets. A route that exists for another method
(GET on POST /v1/chat/completions) still answers 405, as it did before the
catch-all existed. A request that does not ask for HTML -- curl, an SDK, a
script -- gets a 404, never a page: `/` itself stays the JSON service
descriptor for them (mcp/server.py `root`).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

# Module-level because `from __future__ import annotations` makes FastAPI
# resolve the handlers' `request: Request` against THIS module's globals; a
# local import inside mount() turns it into a query parameter and every
# request a 422. The checker (`__main__`) needs no web framework.
try:
    from starlette.requests import Request
except ImportError:  # pragma: no cover
    Request = None  # type: ignore[assignment,misc]

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.normpath(os.environ.get("YAMADORI_DASH_DIST")
                        or os.path.join(HERE, "..", "web", "dist"))
VITE_URL = os.environ.get("YAMADORI_DASH_VITE", "http://localhost:5173/")

# Path prefixes the SPA fallback never answers: they belong to the server.
RESERVED = ("v1", "health", "dash/api", "dash/classic", "assets", "openapi.json")

# web/src files that never feed the bundle, so their absence from .buildinfo
# is expected rather than suspicious.
_NOT_BUNDLED = (".test.ts", ".test.tsx")


def hash_file(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def aggregate(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(files):
        h.update(f"{k}\0{files[k]}\n".encode("utf-8"))
    return h.hexdigest()


def check_buildinfo(dist: str = DIST) -> dict:
    """Does the bundle in `dist` match the source beside it?

    Returns {"ok", "problems", "checked", "unlisted"}. `problems` are
    failures, each naming a file and what is wrong with it. `unlisted` are
    source files the build did not read: not a failure (an unimported file
    cannot change the bundle), but listed so a forgotten import is visible.
    """
    root = os.path.dirname(os.path.abspath(dist))
    info_path = os.path.join(dist, ".buildinfo")
    out: dict = {"ok": False, "problems": [], "checked": 0, "unlisted": []}
    if not os.path.exists(os.path.join(dist, "index.html")):
        out["problems"].append(f"{dist} has no index.html: nothing is built")
        return out
    try:
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError) as e:
        out["problems"].append(f"{info_path} unreadable ({type(e).__name__}: {e}); "
                               "rebuild with `npm run build` in web/")
        return out
    for key, want in (("version", 1), ("algorithm", "sha256"),
                      ("normalization", "crlf-to-lf")):
        if info.get(key) != want:
            out["problems"].append(f".buildinfo {key} is {info.get(key)!r}, "
                                   f"this checker understands {want!r}")
    files = info.get("files") or {}
    if not files:
        out["problems"].append(".buildinfo lists no source files")
    if aggregate(files) != info.get("aggregate"):
        out["problems"].append(".buildinfo aggregate does not match its own "
                               "file list: the file was edited by hand")
    for rel, want in sorted(files.items()):
        path = os.path.join(root, *rel.split("/"))
        if not os.path.exists(path):
            out["problems"].append(f"{rel} fed the build and no longer exists")
            continue
        out["checked"] += 1
        if hash_file(path) != want:
            out["problems"].append(f"{rel} changed since the bundle was built")
    src = os.path.join(root, "src")
    if os.path.isdir(src):
        for dirpath, dirnames, names in os.walk(src):
            dirnames[:] = [d for d in dirnames if d != "__fixtures__"]
            for name in names:
                if not name.endswith((".ts", ".tsx", ".css")) or name.endswith(_NOT_BUNDLED):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), root).replace("\\", "/")
                if rel not in files:
                    out["unlisted"].append(rel)
    out["ok"] = not out["problems"]
    return out


def ui_mode(argv: list[str] | None = None) -> str:
    argv = sys.argv if argv is None else argv
    forced = os.environ.get("YAMADORI_DASH_UI", "").strip().lower()
    if forced in ("react", "python"):
        return forced
    if "--dev" in argv or os.environ.get("YAMADORI_DASH_DEV") == "1":
        return "dev"
    return "react" if os.path.exists(os.path.join(DIST, "index.html")) else "python"


def reserved(path: str) -> bool:
    """Is `path` under a prefix the server owns (never the SPA's)?"""
    p = path.strip("/").lower()
    return any(p == r or p.startswith(r + "/") for r in RESERVED)


def wants_html(accept: str | None) -> bool:
    """A browser navigating asks for text/html; curl, SDKs and fetch() of
    JSON do not. Only the former ever gets the SPA."""
    return "text/html" in (accept or "").lower()


def _top_files(dist: str) -> set[str]:
    try:
        return {n for n in os.listdir(dist) if not n.startswith(".") and n != "index.html"
                and os.path.isfile(os.path.join(dist, n))}
    except OSError:
        return set()


def _dev_page() -> str:
    return ("<!doctype html><meta charset=utf-8><title>Yamadori (dev)</title>"
            "<body style='background:#0a0e17;color:#dfe2ef;font:14px monospace;"
            "padding:2rem'><p>Dashboard dev mode: the React build is not served "
            f"here.</p><p>Open <a style='color:#5cf2ff' href='{VITE_URL}'>{VITE_URL}</a>"
            " (Vite, with HMR). It proxies /dash/api to this server.</p>")


def index_response(mode: str, dist: str = DIST, build: str = "ok"):
    """The SPA's HTML entry: index.html (react) or the pointer to Vite (dev).
    index.html is never cached: it is the one file whose name does not change
    when the build does."""
    from fastapi.responses import FileResponse, HTMLResponse
    if mode == "dev":
        return HTMLResponse(_dev_page(), headers={"Cache-Control": "no-store", "Vary": "Accept"})
    return FileResponse(os.path.join(dist, "index.html"), media_type="text/html",
                        headers={"Cache-Control": "no-cache", "Vary": "Accept",
                                 "X-Yamadori-Dash-Build": build})


def mount(app, mode: str, dist: str = DIST) -> dict:
    """Register the SPA on `app`. Call AFTER every other route: Starlette
    matches in registration order and the fallback is a catch-all, so a GET
    route registered after it is unreachable.
    Returns the buildinfo check (or a note, in dev) for the caller to log."""
    from fastapi.responses import FileResponse, RedirectResponse
    from starlette.exceptions import HTTPException
    from starlette.routing import Match
    from starlette.staticfiles import StaticFiles

    if mode == "dev":
        check = {"ok": True, "problems": [], "note": f"dev mode, UI at {VITE_URL}"}
        build = "dev"
    else:
        check = check_buildinfo(dist)
        build = "ok" if check["ok"] else "stale"

        class ImmutableAssets(StaticFiles):
            # File names are content-hashed by Vite, so a name never changes
            # meaning: cache forever.
            async def get_response(self, path, scope):
                resp = await super().get_response(path, scope)
                if resp.status_code == 200:
                    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                return resp

        app.mount("/assets", ImmutableAssets(directory=os.path.join(dist, "assets")),
                  name="dash-assets")

    async def dash_moved(request: Request, rest: str = ""):
        # The dashboard moved to the site root; old bookmarks follow it.
        # /dash/api/* and /dash/classic* are registered earlier and never
        # arrive here; if one does (a misordered registration) it is a 404,
        # never a redirect into the SPA.
        if reserved("dash/" + rest):
            raise HTTPException(status_code=404, detail="Not Found")
        q = request.url.query
        return RedirectResponse("/" + rest + (f"?{q}" if q else ""), status_code=302)

    async def spa(request: Request, rest: str = ""):
        # A path another route serves by a different method keeps its 405.
        allow: set[str] = set()
        for route in request.app.router.routes:
            if getattr(route, "endpoint", None) is spa:
                continue
            match, _ = route.matches(request.scope)
            if match == Match.PARTIAL:
                allow |= set(getattr(route, "methods", None) or ())
        if allow:
            raise HTTPException(status_code=405, detail="Method Not Allowed",
                                headers={"Allow": ", ".join(sorted(allow))})
        if reserved(rest):
            raise HTTPException(status_code=404, detail="Not Found")
        # A real top-level file Vite emitted (favicon and the like). Never a
        # dotfile: .buildinfo is for the checker, not the browser.
        # Matched by name against the directory listing, so no path the
        # client sends is ever joined onto the filesystem.
        if mode != "dev" and rest and rest in _top_files(dist):
            return FileResponse(os.path.join(dist, rest))
        if not wants_html(request.headers.get("accept")):
            raise HTTPException(status_code=404, detail="Not Found")
        return index_response(mode, dist, build)

    app.add_api_route("/dash", dash_moved, methods=["GET"], include_in_schema=False)
    app.add_api_route("/dash/{rest:path}", dash_moved, methods=["GET"], include_in_schema=False)
    app.add_api_route("/{rest:path}", spa, methods=["GET"], include_in_schema=False)
    return check


if __name__ == "__main__":
    c = check_buildinfo()
    print(f"  {DIST}")
    print(f"  {c['checked']} source files checked, "
          f"{'bundle is CURRENT' if c['ok'] else 'bundle is STALE'}")
    for p in c["problems"]:
        print(f"    FAIL  {p}")
    for u in c["unlisted"]:
        print(f"    note  {u} did not feed the build")
    sys.exit(0 if c["ok"] else 1)
