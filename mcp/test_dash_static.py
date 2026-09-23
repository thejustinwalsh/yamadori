#!/usr/bin/env python
"""The committed dashboard bundle, asserted. No Node, no GPU, no server.

WHAT THIS IS GATING

`web/dist` is committed so the stack runs without Node. The failure that
invites is a bundle that no longer matches its source: someone edits a .tsx,
does not rebuild, commits, and the dashboard silently is not what the code
says. `web/dist/.buildinfo` records a sha256 of every source file that fed the
build; this recomputes them in Python and fails if any differs.

It also asserts the serving shape `dash_static.mount()` gives the SPA, which
owns the site root:

  - /, /nebari, /data/<id> ... get index.html, uncached, when the request
    asks for HTML; a request that does not (curl, an SDK) gets a 404
  - / stays the JSON service descriptor for clients that do not ask for HTML
  - /dash/api/*, /dash/classic*, /v1/*, /health still reach their routes,
    never index.html, and a GET on a POST-only route is still a 405
  - old /dash and /dash/<route> bookmarks 302 to the root equivalent
  - a missing /assets/*.js is a 404, never index.html (an HTML body served
    for a JS chunk fails as a MIME error nobody can read)
  - hashed assets are cached immutable
  - dev mode never serves the bundle, and says where Vite is

and that the checker itself notices what it exists to notice: an edited file,
a deleted file, a hand-edited .buildinfo -- and does NOT fire on a CRLF-only
difference, which would make it cry wolf on every Windows checkout.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dash_static  # noqa: E402

try:  # module-level: `from __future__ import annotations` resolves handler
    from starlette.requests import Request  # annotations against these globals
except ImportError:  # pragma: no cover
    Request = None  # type: ignore[assignment,misc]

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def _mini_root() -> str:
    """A throwaway web/ with two sources, a dist, and a .buildinfo written by
    the same rules the Node writer uses."""
    root = tempfile.mkdtemp(prefix="yamadori_test_dash_")
    os.makedirs(os.path.join(root, "src"))
    os.makedirs(os.path.join(root, "dist", "assets"))
    files = {"src/App.tsx": b"export const a = 1;\n", "package.json": b"{}\n"}
    for rel, body in files.items():
        with open(os.path.join(root, *rel.split("/")), "wb") as f:
            f.write(body)
    with open(os.path.join(root, "dist", "index.html"), "w") as f:
        f.write("<!doctype html><div id=root></div>")
    with open(os.path.join(root, "dist", "assets", "index-abc123.js"), "w") as f:
        f.write("console.log(1)")
    hashes = {rel: dash_static.hash_file(os.path.join(root, *rel.split("/"))) for rel in files}
    info = {"version": 1, "algorithm": "sha256", "normalization": "crlf-to-lf", "root": "web",
            "files": hashes, "aggregate": dash_static.aggregate(hashes)}
    with open(os.path.join(root, "dist", ".buildinfo"), "w") as f:
        json.dump(info, f)
    return root


def test_the_committed_bundle_matches_its_source():
    c = dash_static.check_buildinfo()
    check(c["ok"], "web/dist matches web/src (rebuild with `npm run build` in web/ if not)",
          "; ".join(c["problems"][:5]))
    check(c["checked"] >= 20, f"it actually hashed the sources ({c['checked']} files)",
          "a check over zero files is not a check")
    info = json.load(open(os.path.join(dash_static.DIST, ".buildinfo"), encoding="utf-8"))
    check("src/App.tsx" in info["files"], ".buildinfo lists the app entry, i.e. came from the module graph")
    check("package-lock.json" in info["files"], ".buildinfo covers the lockfile, so a dependency bump is a source change")
    check(not any(k.endswith((".test.ts", ".test.tsx")) for k in info["files"]),
          "test files are not recorded as build inputs")


def test_the_checker_notices_what_it_exists_to_notice():
    root = _mini_root()
    dist = os.path.join(root, "dist")
    try:
        check(dash_static.check_buildinfo(dist)["ok"], "a freshly written .buildinfo passes")

        app = os.path.join(root, "src", "App.tsx")
        with open(app, "wb") as f:
            f.write(b"export const a = 1;\r\n")
        check(dash_static.check_buildinfo(dist)["ok"], "a CRLF-only difference is NOT reported stale")

        with open(app, "wb") as f:
            f.write(b"export const a = 2;\n")
        c = dash_static.check_buildinfo(dist)
        check(not c["ok"] and any("src/App.tsx changed" in p for p in c["problems"]),
              "an edited source file is reported by name", str(c["problems"]))

        os.remove(app)
        c = dash_static.check_buildinfo(dist)
        check(not c["ok"] and any("no longer exists" in p for p in c["problems"]),
              "a deleted source file is reported", str(c["problems"]))

        root2 = _mini_root()
        info_path = os.path.join(root2, "dist", ".buildinfo")
        info = json.load(open(info_path))
        info["files"]["src/App.tsx"] = "0" * 64
        json.dump(info, open(info_path, "w"))
        c = dash_static.check_buildinfo(os.path.join(root2, "dist"))
        check(any("edited by hand" in p for p in c["problems"]),
              "a hand-edited .buildinfo fails its own aggregate", str(c["problems"]))
        shutil.rmtree(root2, ignore_errors=True)

        with open(os.path.join(root, "src", "Orphan.tsx"), "w") as f:
            f.write("export {}")
        c = dash_static.check_buildinfo(dist)
        check("src/Orphan.tsx" in c["unlisted"], "a source file the build never read is listed, not failed")

        os.remove(os.path.join(dist, ".buildinfo"))
        check(not dash_static.check_buildinfo(dist)["ok"], "a bundle with no .buildinfo fails")
    finally:
        shutil.rmtree(root, ignore_errors=True)


HTML = {"accept": "text/html,application/xhtml+xml,*/*;q=0.8"}


def _app_like_server(dist: str, mode: str = "react"):
    """The routes mcp/server.py registers before the SPA, in its order, with
    stub bodies: what matters here is who answers, not what they say."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()
    state: dict = {}

    @app.get("/")
    async def root(request: Request):
        if dash_static.wants_html(request.headers.get("accept")):
            return dash_static.index_response(mode, dist)
        return JSONResponse({"service": "yamadori"})

    @app.get("/health")
    async def health():
        return JSONResponse({"ok": True})

    @app.get("/v1/models")
    async def models():
        return JSONResponse({"data": []})

    @app.get("/dash/classic")
    async def classic():
        return JSONResponse({"classic": True})

    @app.get("/dash/api/{rest:path}")
    async def api(rest: str):
        return JSONResponse({"api": rest})

    @app.post("/dash/api/{rest:path}")
    async def api_post(rest: str):
        return JSONResponse({"posted": rest})

    @app.post("/v1/chat/completions")
    async def chat():
        return JSONResponse({"chat": True})

    state["check"] = dash_static.mount(app, mode, dist)
    return app, state


def test_the_spa_owns_the_root_and_never_shadows_the_server():
    from starlette.testclient import TestClient

    root = _mini_root()
    try:
        app, state = _app_like_server(os.path.join(root, "dist"))
        check(state["check"]["ok"], "mount() returns the buildinfo check")
        client = TestClient(app)

        for path in ("/", "/nebari", "/data", "/data/some-id", "/results", "/phase0", "/no-such-screen"):
            r = client.get(path, headers=HTML)
            check(r.status_code == 200 and "text/html" in r.headers["content-type"]
                  and "id=root" in r.text, f"{path} (Accept: text/html) serves index.html", f"{r.status_code}")
        check(client.get("/nebari", headers=HTML).headers.get("cache-control") == "no-cache",
              "index.html is served no-cache")

        r = client.get("/", headers={"accept": "application/json"})
        check(r.status_code == 200 and r.json() == {"service": "yamadori"},
              "/ with Accept: application/json is still the JSON descriptor")
        r = client.get("/", headers={"accept": "*/*"})
        check(r.status_code == 200 and r.json() == {"service": "yamadori"},
              "/ with curl's Accept: */* is still the JSON descriptor")
        r = client.get("/nebari", headers={"accept": "application/json"})
        check(r.status_code == 404 and "id=root" not in r.text,
              "a client route asked for as JSON is a 404, never the page", f"{r.status_code}")

        r = client.get("/dash/api/vitals", headers=HTML)
        check(r.status_code == 200 and r.json() == {"api": "vitals"}, "/dash/api/* still reaches the API")
        r = client.post("/dash/api/dataset", headers=HTML)
        check(r.status_code == 200 and r.json() == {"posted": "dataset"}, "POST /dash/api/* still reaches the API")
        r = client.get("/dash/classic", headers=HTML)
        check(r.status_code == 200 and r.json() == {"classic": True}, "/dash/classic still reaches the Python page")
        check(client.get("/health", headers=HTML).json() == {"ok": True}, "/health is untouched")
        check(client.get("/v1/models", headers=HTML).json() == {"data": []}, "/v1/models is untouched")
        r = client.get("/v1/chat/completions", headers=HTML)
        check(r.status_code == 405 and "POST" in r.headers.get("allow", ""),
              "GET on POST-only /v1/chat/completions is still a 405, not the page", f"{r.status_code}")
        for path in ("/v1/nope", "/dash/api", "/dash/classic/nope", "/assets", "/health/x"):
            r = client.get(path, headers=HTML)
            check(r.status_code in (404, 405) and "id=root" not in r.text,
                  f"{path} is never answered by the SPA", f"{r.status_code}")

        for old, new in (("/dash", "/"), ("/dash/", "/"), ("/dash/nebari", "/nebari"),
                         ("/dash/data/some-id", "/data/some-id"), ("/dash/results?x=1", "/results?x=1")):
            r = client.get(old, headers=HTML, follow_redirects=False)
            check(r.status_code == 302 and r.headers.get("location") == new,
                  f"{old} redirects to {new}", f"{r.status_code} {r.headers.get('location')}")

        r = client.get("/assets/index-abc123.js")
        check(r.status_code == 200 and "immutable" in r.headers.get("cache-control", ""),
              "hashed assets are served from /assets and cached immutable")
        r = client.get("/assets/missing-chunk.js", headers=HTML)
        check(r.status_code == 404 and "id=root" not in r.text,
              "a missing asset is a 404, never index.html", f"{r.status_code}")
        r = client.get("/.buildinfo", headers={"accept": "*/*"})
        check(r.status_code == 404, "dist/.buildinfo itself is not served", f"{r.status_code}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_dev_mode_never_serves_the_bundle():
    from starlette.testclient import TestClient

    root = _mini_root()
    try:
        app, _ = _app_like_server(os.path.join(root, "dist"), mode="dev")
        client = TestClient(app)
        for path in ("/", "/nebari"):
            r = client.get(path, headers=HTML)
            check(r.status_code == 200 and dash_static.VITE_URL in r.text and "id=root" not in r.text,
                  f"dev mode: {path} points at Vite and serves no bundle")
        check(client.get("/assets/index-abc123.js").status_code == 404,
              "dev mode serves no assets")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_mode_selection():
    old = {k: os.environ.pop(k, None) for k in ("YAMADORI_DASH_UI", "YAMADORI_DASH_DEV")}
    try:
        check(dash_static.ui_mode(["server.py", "--dev"]) == "dev", "--dev selects dev mode")
        os.environ["YAMADORI_DASH_UI"] = "python"
        check(dash_static.ui_mode(["server.py", "--dev"]) == "python", "YAMADORI_DASH_UI=python always wins")
        os.environ.pop("YAMADORI_DASH_UI")
        check(dash_static.ui_mode(["server.py"]) == "react", "a built bundle selects react by default")
    finally:
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


def main() -> int:
    for fn in (test_the_committed_bundle_matches_its_source,
               test_the_checker_notices_what_it_exists_to_notice,
               test_the_spa_owns_the_root_and_never_shadows_the_server,
               test_dev_mode_never_serves_the_bundle,
               test_mode_selection):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    if passed < total:
        print("  If only the first check failed: web/dist is stale. Rebuild it.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
