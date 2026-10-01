#!/usr/bin/env python
"""TypeSafe's own SDKs against the Jev API (mcp/jev_api.py) -- the conformance
test docs/JEV-CONFORMANCE.md section 6 planned.

    python mcp/test_jev_sdk.py            OFFLINE: a local server of
                                          server.app on 127.0.0.1:<free port>,
                                          the stubbed decider of
                                          mcp/test_jev_api.py behind it
    YAMADORI_TEST_KEY=... python mcp/test_jev_sdk.py --live
                                          LIVE: the proxy's /jev routes
                                          (YAMADORI_PROXY, default :1234);
                                          NOT RUN until a deploy has them

THE CLIENTS: typesafe-sdk 0.7.2 in tools/typesafe-sdk/py/.venv and
@typesafe-ai/sdk 0.6.0 in tools/typesafe-sdk/js/node_modules, installed by
tools/typesafe-sdk/install.py --run (pinned; models/manifest.yaml runtimes
typesafe-sdk-python / typesafe-sdk-js). Each runs mcp/jev_sdk/driver.py /
driver.mjs as a subprocess; this file judges the JSON lines they print.
scripts/run_tests.py skips this suite, with a note, until both are installed
(REQUIRES); run directly without them, it fails on "installed".

WHAT IS CHECKED, per SDK:
  1. models.list() parses (Python: ListModelsResponse of ModelMetadata under
     the SDK's strict models).
  2. all 13 doc examples (mcp/fixtures/jev_doc_examples.json) parse as the
     SDK's response (Python: SystemOneResponse, NoulAnswer / ChoiceAnswer /
     ScoreAnswer by type, .nouls / .choices / .scores) -- offline also with
     the docs' values (the fake reads each example's probabilities).
  3. a 422 raises the SDK's validation error, and its message names the
     field (questions.q.criteria).
  4. a 429 carries retry-after 1000 ms (retries off), and with the default
     policy is retried and succeeds; a 529 is retried and succeeds (offline:
     the harness makes the first attempt fail; the server's log shows both).
  5. request_id is the x-typesafe-request-id header's value, and (offline)
     the id of a corpus jev_call row.
Before the SDKs, the HARNESS itself is checked over plain HTTP (the scenario
header gives each refusal it promises), so an SDK failure is the SDK's.

The key is passed to the drivers in TYPESAFE_API_KEY and never printed.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
LIVE = "--live" in sys.argv

SDK_DIR = os.path.join(ROOT, "tools", "typesafe-sdk")
PY = os.path.join(SDK_DIR, "py", ".venv", "Scripts" if os.name == "nt" else "bin",
                  "python.exe" if os.name == "nt" else "python")
JS_PKG = os.path.join(SDK_DIR, "js", "node_modules", "@typesafe-ai", "sdk",
                      "package.json")
PINS = {"python": "0.7.2", "js": "0.6.0"}
DRIVERS = {"python": os.path.join(HERE, "jev_sdk", "driver.py"),
           "js": os.path.join(HERE, "jev_sdk", "driver.mjs")}
EXAMPLES = os.path.join(HERE, "fixtures", "jev_doc_examples.json")

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail="") -> bool:
    _results.append((bool(ok), name, str(detail)[:600]))
    return bool(ok)


def installed() -> dict:
    """{python: bool, js: bool, node: path | None}."""
    node = None
    for n in ("node.exe", "node"):
        for d in os.environ.get("PATH", "").split(os.pathsep):
            p = os.path.join(d, n)
            if os.path.isfile(p):
                node = p
                break
        if node:
            break
    return {"python": os.path.isfile(PY), "js": os.path.isfile(JS_PKG) and bool(node),
            "node": node}


with open(EXAMPLES, encoding="utf-8") as _f:
    EX = json.load(_f)["examples"]


# ----------------------------------------------------------- judgement ------
def same_values(got: dict, want: dict) -> tuple[bool, str]:
    """An SDK's parsed answer against the doc's (as mcp/test_jev_api.py
    judges the wire): exact where the fake is exact, confidence and score
    within the rounding of the docs' two decimals."""
    t = want["type"]
    if got.get("type") != t:
        return False, f"type {got.get('type')} vs {t}"
    if t == "noul":
        return abs(got["noul"] - want["noul"]) <= 1e-6, "noul"
    n = len(want["probabilities"])
    if set(got["probabilities"]) != set(want["probabilities"]):
        return False, "probability keys"
    if any(abs(got["probabilities"][k] - v) > 1e-6
           for k, v in want["probabilities"].items()):
        return False, "probabilities"
    if abs(got["confidence"] - want["confidence"]) > n / (n - 1) * 0.005 + 0.005:
        return False, "confidence"
    if t == "choice":
        return got["choice"] == want["choice"], "choice"
    ok = abs(got["score"] - want["score"]) <= 0.005 * sum(range(n)) + 0.005
    return ok and got["legend"] == want["legend"], "score / legend"


CLASS_OF = {"noul": "NoulAnswer", "choice": "ChoiceAnswer", "score": "ScoreAnswer"}


def judge(sdk: str, lines: list[dict], *, live: bool, log=None,
          corpus_ids: set | None = None) -> None:
    tag = f"[{sdk}]"
    by = {r.get("step"): r for r in lines}
    s = by.get("sdk") or {}
    check(s.get("version") == PINS[sdk], f"{tag} the pinned SDK ran "
          f"({PINS[sdk]})", s)
    check("done" in by, f"{tag} the driver ran to the end", list(by)[-3:])
    rids = []
    m = by.get("models") or {}
    ok = m.get("ok") and m.get("models") and m["models"][0]["name"] == "jjava-latest" \
        and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", x.get("release_date") or "")
                for x in m["models"])
    if sdk == "python":
        ok = ok and m.get("class") == "ListModelsResponse" and all(
            x.get("class") == "ModelMetadata" for x in m["models"])
    check(bool(ok), f"{tag} models.list() parses", m)
    check(bool(m.get("request_id")) and m.get("request_id") == m.get("header_request_id"),
          f"{tag} models: request_id is the x-typesafe-request-id header", m)
    rids.append(m.get("request_id"))
    for i, e in enumerate(EX):
        r = by.get(f"example:{i}") or {}
        want = e["response"]["answers"]
        label = f"{tag} doc example {i + 1} ({e['source'].rsplit('/', 1)[-1]})"
        if not check(bool(r.get("ok")), f"{label} parses", r):
            continue
        got = r.get("answers") or {}
        ok, why = set(got) == set(want), "answer ids"
        for qid, w in want.items():
            if not ok:
                break
            g = got[qid]
            if sdk == "python" and g.get("_class") != CLASS_OF[w["type"]]:
                ok, why = False, f"{qid}: {g.get('_class')}"
                break
            if not set(w) <= set(g):
                ok, why = False, f"{qid}: fields {sorted(g)}"
                break
            if not live:
                ok, why = same_values(g, w)
                why = f"{qid}: {why}"
        if sdk == "python":
            u = r.get("usage") or {}
            ok = ok and r.get("class") == "SystemOneResponse" and \
                u.get("class") == "Usage" and isinstance(u.get("input_tokens"), int)
            split = r.get("split") or {}
            ok = ok and sorted(split.get("nouls", []) + split.get("choices", [])
                               + split.get("scores", [])) == sorted(want)
        check(ok, f"{label}: the SDK's response{' with the docs values' if not live else ''}",
              f"{why}: {json.dumps(r)[:400]}")
        check(bool(r.get("request_id")) and r["request_id"] == r.get("header_request_id"),
              f"{label}: request_id is the header's", r.get("request_id"))
        rids.append(r.get("request_id"))
    v = by.get("invalid_422") or {}
    cls = ("TypeSafeUnprocessableEntityError" if sdk == "python"
           else "UnprocessableEntityError")
    check(v.get("ok") is False and v.get("error_class") == cls and v.get("status") == 422,
          f"{tag} a 422 raises {cls}", v)
    check("criteria" in (v.get("message") or "") and "q" in (v.get("message") or ""),
          f"{tag} the 422's message names the field (questions.q.criteria)",
          v.get("message"))
    if live:
        return
    b = by.get("busy_no_retry") or {}
    rl = "TypeSafeRateLimitError" if sdk == "python" else "RateLimitError"
    check(b.get("ok") is False and b.get("error_class") == rl and b.get("status") == 429,
          f"{tag} a 429 (retries off) raises {rl}", b)
    check(b.get("retry_after_ms") == 1000, f"{tag} the 429's retry-after is 1000 ms",
          b.get("retry_after_ms"))
    for name, first in (("busy_retried", 429), ("overload_retried", 529)):
        r = by.get(name) or {}
        prefix = "busy_once" if first == 429 else "overload_once"
        seen = [x for x in (log or []) if x["scenario"].startswith(f"{prefix}:"
                                                                   f"{'py' if sdk == 'python' else 'js'}:")]
        ok = r.get("ok") and [x["status"] for x in seen] == [first, 200]
        gap = seen[1]["t"] - seen[0]["t"] if len(seen) == 2 else None
        check(bool(ok) and gap is not None and gap >= 0.9,
              f"{tag} a {first} is retried by the default policy after its "
              f"Retry-After (1 s), then succeeds",
              {"step": r, "attempts": seen, "gap_s": gap})
        rids.append(r.get("request_id"))
    if corpus_ids is not None:
        missing = [x for x in rids if x and x not in corpus_ids]
        check(not missing and all(rids), f"{tag} every request_id is a corpus "
              "jev_call row's id", missing)


def run_driver(sdk: str, base: str, key: str, plan: dict, node: str | None) -> list[dict]:
    fd, path = tempfile.mkstemp(prefix="jev_sdk_plan_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(plan, f)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TYPESAFE_") and k != "YAMADORI_TEST_KEY"}
    env["TYPESAFE_API_KEY"] = key
    if sdk == "python":
        cmd = [PY, DRIVERS["python"], "--base-url", base, "--plan", path]
    else:
        cmd = [node, DRIVERS["js"], "--base-url", base, "--plan", path,
               "--sdk-dir", os.path.join(SDK_DIR, "js")]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding="utf-8", timeout=600)
    finally:
        os.remove(path)
    lines = []
    for ln in r.stdout.splitlines():
        try:
            lines.append(json.loads(ln))
        except ValueError:
            pass
    if r.returncode != 0 or not lines:
        lines.append({"step": "driver_exit", "rc": r.returncode,
                      "stderr": r.stderr.replace(key, "***")[-800:]})
    return lines


def need_sdks(have: dict) -> bool:
    ok = check(have["python"], "the TypeSafe Python SDK is installed "
               "(tools/typesafe-sdk/py/.venv)",
               "run: python tools/typesafe-sdk/install.py --run")
    ok &= check(have["js"], "the TypeSafe JS SDK and node are installed "
                "(tools/typesafe-sdk/js/node_modules)",
                "run: python tools/typesafe-sdk/install.py --run")
    return ok


# ------------------------------------------------------------- offline ------
def offline() -> None:
    import test_jev_api as T       # its temp stores, the fake, the app, a key
    import jev_api as J
    import max_mode
    # Test-only: the 529 path's Retry-After (max_mode.RETRY_AFTER_UNKNOWN,
    # 30 s) set to 1 s, so a retried 529 takes a second, not half a minute.
    max_mode.RETRY_AFTER_UNKNOWN = 1

    class Scenarios:
        """An ASGI wrapper over server.app: the header x-jev-test picks what
        the next request meets -- a doc example's probabilities, a busy lane
        (429), an unreachable model server (529) -- once or always. Every
        request is logged with its status."""

        def __init__(self, app):
            self.app, self.log, self.attempts = app, [], {}

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await self.app(scope, receive, send)
            sc = dict(scope.get("headers") or []).get(b"x-jev-test", b"").decode()
            n = self.attempts[sc] = self.attempts.get(sc, 0) + 1
            hold = False
            T.FAKE.reset()
            if sc.startswith("example:"):
                T.FAKE.reset(T.weights_for(EX[int(sc.split(":")[1])]))
            elif sc == "busy_always" or (sc.startswith("busy_once:") and n == 1):
                hold = J._LANE.acquire(blocking=False)
            elif sc.startswith("overload_once:") and n == 1:
                T.FAKE.fail_post = urllib.error.URLError("connection refused")
            rec = {"scenario": sc, "attempt": n, "t": time.time(), "status": None}
            self.log.append(rec)

            async def send2(msg):
                if msg.get("type") == "http.response.start":
                    rec["status"] = msg.get("status")
                await send(msg)
            try:
                await self.app(scope, receive, send2)
            finally:
                if hold:
                    J._LANE.release()

    import uvicorn
    import server
    scen = Scenarios(server.app)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = uvicorn.Server(uvicorn.Config(scen, host="127.0.0.1", port=port,
                                        log_level="warning", lifespan="off"))
    threading.Thread(target=srv.run, daemon=True).start()
    t0 = time.time()
    while not srv.started and time.time() - t0 < 20:
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}/jev"
    check(srv.started, "the local test server is up (127.0.0.1, a free port)", base)

    # --- the harness itself, over plain HTTP
    def call(sc: str, body: dict):
        req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {T.KEY}",
                                              "Content-Type": "application/json",
                                              "x-jev-test": sc})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, json.loads(e.read() or b"null")
    st, h, d = call("example:0", EX[0]["request"])
    check(st == 200 and same_values(d["answers"]["is_urgent"],
                                    EX[0]["response"]["answers"]["is_urgent"])[0],
          "harness: a doc example's scenario gives the docs' answer", (st, d))
    st, h, d = call("busy_always", EX[0]["request"])
    check(st == 429 and h.get("retry-after") == "1",
          "harness: busy_always is a 429 with Retry-After 1", (st, h))
    sc = f"overload_once:harness:{time.time()}"
    st1, h1, _ = call(sc, EX[0]["request"])
    st2, _, _ = call(sc, EX[0]["request"])
    check(st1 == 529 and h1.get("retry-after") == "1" and st2 == 200,
          "harness: overload_once is a 529 (Retry-After 1), then 200", (st1, h1, st2))

    have = installed()
    if not need_sdks(have):
        return
    plan = {"live": False, "examples": EX}
    for sdk in ("python", "js"):
        lines = run_driver(sdk, base, T.KEY, plan, have["node"])
        con = sqlite3.connect(os.environ["YAMADORI_CORPUS_DB"])
        ids = {t for (t,) in con.execute(
            "SELECT turn FROM events WHERE kind='jev_call'")}
        con.close()
        judge(sdk, lines, live=False, log=scen.log, corpus_ids=ids)
        if any(r.get("step") == "driver_exit" for r in lines):
            check(False, f"[{sdk}] the driver exited cleanly",
                  [r for r in lines if r.get("step") == "driver_exit"])
    srv.should_exit = True


# ---------------------------------------------------------------- live ------
def live() -> int:
    key = os.environ.get("YAMADORI_TEST_KEY", "").strip()
    proxy = os.environ.get("YAMADORI_PROXY", "http://127.0.0.1:1234").rstrip("/")
    base = proxy + "/jev"
    if not key:
        print("  1 test NOT RUN (no key: set YAMADORI_TEST_KEY)")
        print("\n  0/0 live checks passed")
        return 3
    req = urllib.request.Request(base + "/v1/models",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            st = r.status
    except urllib.error.HTTPError as e:
        st = e.code
    except OSError as e:
        print(f"  the proxy is unreachable at {proxy}: {type(e).__name__}")
        print("\n  0/1 live checks passed")
        return 1
    if st == 404:
        print("  1 test NOT RUN (not deployed): GET /jev/v1/models is 404 -- "
              "the /jev routes reach the proxy at the next deploy")
        print("\n  0/0 live checks passed")
        return 3
    if st == 429:
        print("  1 test NOT RUN (429)")
        print("\n  0/0 live checks passed")
        return 3
    check(st == 200, "GET /jev/v1/models answers", st)
    have = installed()
    if need_sdks(have):
        for sdk in ("python", "js"):
            judge(sdk, run_driver(sdk, base, key, {"live": True, "examples": EX},
                                  have["node"]), live=True)
    return report("live ")


def report(kind: str = "") -> int:
    for ok, name, detail in _results:
        print(("  pass  " if ok else "  FAIL  ") + name
              + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} {kind}checks passed")
    return 0 if passed == len(_results) else 1


def main() -> int:
    if LIVE:
        return live()
    offline()
    return report()


if __name__ == "__main__":
    sys.exit(main())
